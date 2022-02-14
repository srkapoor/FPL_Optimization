import pandas as pd
import numpy as np
import os
import requests
from pulp import *

def get_data(team_id, gw):
    # get FPL api data
    r = requests.get('https://fantasy.premierleague.com/api/bootstrap-static/')
    fpl_data = r.json()

    # get FPL player and team data
    element_data = pd.DataFrame(fpl_data['elements'])
    team_data = pd.DataFrame(fpl_data['teams'])
    type_data = pd.DataFrame(fpl_data['element_types']).set_index(['id'])
    elements_team = pd.merge(element_data, team_data, left_on='team', right_on='id')

    # get fplreview data and create final merged table
    review_data =  pd.read_csv('data/fplreview_gw26_33.csv')
    review_data['review_id'] = review_data.index+1
    merged_data = pd.merge(elements_team, review_data, left_on='id_x', right_on='review_id')
    merged_data.set_index(['id_x'], inplace=True)
    next_gw = int(review_data.keys()[5].split('_')[0])

    # get my team data
    r = requests.get(f'https://fantasy.premierleague.com/api/entry/{team_id}/event/{gw}/picks/')
    picks_data = r.json()
    initial_squad = [i['element'] for i in picks_data['picks']]
    r = requests.get(f'https://fantasy.premierleague.com/api/entry/{team_id}/')
    general_data = r.json()
    itb = general_data['last_deadline_bank'] / 10

    return {
        'merged_data': merged_data,
        'elements_team': elements_team,
        'team_data' : team_data,
        'type_data' : type_data,
        'next_gw' : next_gw,
        'initial_squad' : initial_squad,
        'review_data': review_data,
        'itb': itb
    }

def get_transfer_history(team_id, last_gw):
    transfers = []
    # Reversing GW history until a chip is played or 2+ transfers were made
    for gw in range(last_gw, 0, -1):
        res = requests.get(f'https://fantasy.premierleague.com/api/entry/{team_id}/event/{gw}/picks/').json()
        transfer = res['entry_history']['event_transfers']
        chip = res['active_chip']

        transfers.append(transfer)
        if transfer > 1 or (chip is not None and (chip != '3xc' or chip != 'bboost')):
            break

    return transfers

def get_rolling(team_id, last_gw):
    transfers = get_transfer_history(team_id, last_gw)

    # Start from gw where last chip used or when hits were taken
    # Reset FT count
    rolling = 0
    for transfer in reversed(transfers):
        # Transfer logic
        rolling = min(max(rolling + 1 - transfer, 0), 1)

    return rolling, transfers[0]

def multi_period_solver(team_id, gw, horizon, objective='regular', decay_base=0.84, team_override=False, team=[]):
    """
    Solves decision support problem for FPL
    Parameters
    ----------
    team_id: integer
        FPL ID of the team to be optimized
    gw: integer
        Upcoming (next) gameweek
    options: dict
        Options for the FPL problem
    """

    # load data
    data = get_data(team_id, gw-1)
    ft = get_rolling(team_id, gw - 1)[0] + 1
    merged_data = data['merged_data']
    team_data = data['team_data']
    type_data = data['type_data']
    next_gw = gw
    initial_squad = data['initial_squad']
    itb = data['itb']
    problem_name = f'mp_b{itb}_h{horizon}_o{objective}_d{decay_base}'

    # prepare data
    players = merged_data.index.to_list()
    element_types = type_data.index.to_list()
    teams = team_data['name'].to_list()
    gameweeks = list(range(next_gw, next_gw + horizon))
    all_gws = [next_gw-1] + gameweeks
    # initialize solver
    prob = LpProblem(problem_name, LpMaximize)

    order = [0, 1, 2, 3]
    bench_weights = {0: 0.03, 1: 0.21, 2: 0.06, 3: 0.002}

    # set variables
    squad = LpVariable.dicts('squad', (players, all_gws), 0, 1, cat='Binary')
    lineup = LpVariable.dicts('lineup', (players, gameweeks), 0, 1, cat='Binary')
    captain = LpVariable.dicts('captain', (players, gameweeks), 0, 1, cat='Binary')
    vicecap = LpVariable.dicts('vicecap', (players, gameweeks), 0, 1, cat='Binary')
    transfer_in = LpVariable.dicts('transfer_in', (players, gameweeks), 0, 1, cat='Binary')
    transfer_out = LpVariable.dicts('transfer_out', (players, gameweeks), 0, 1, cat='Binary')
    in_the_bank = LpVariable.dicts('itb', all_gws, lowBound=0, cat="Continuous")
    free_transfers = LpVariable.dicts('ft', all_gws, lowBound=1, upBound=2, cat="Integer")
    penalized_transfers = LpVariable.dicts('pt', gameweeks, lowBound=0, cat="Integer")
    aux = LpVariable.dicts('aux', gameweeks, 0, 1, cat='Binary')
    bench = LpVariable.dicts('bench', (players, gameweeks, order), 0, 1, cat = 'Binary')
    # dictionaries
    lineup_type_count = {(t, w): lpSum(lineup[p][w] for p in players if merged_data.loc[p, 'element_type'] == t) for t in element_types for w in gameweeks}
    squad_type_count = {(t, w): lpSum(squad[p][w] for p in players if merged_data.loc[p, 'element_type'] == t) for t in element_types for w in gameweeks}
    player_type = merged_data['element_type'].to_dict()
    player_price = (merged_data['now_cost'] / 10).to_dict()
    sold_amount = {w: lpSum(player_price[p] * transfer_out[p][w] for p in players) for w in gameweeks}
    bought_amount = {w: lpSum(player_price[p] * transfer_in[p][w] for p in players) for w in gameweeks}
    points_player_week = {(p, w): merged_data.loc[p, f'{w}_Pts'] for p in players for w in gameweeks}
    squad_count = {w: lpSum(squad[p][w] for p in players) for w in gameweeks}
    lineup_count = {w: lpSum(lineup[p][w] for p in players) for w in gameweeks}
    captain_count = {w: lpSum(captain[p][w] for p in players) for w in gameweeks}
    vicecap_count = {w: lpSum(vicecap[p][w] for p in players) for w in gameweeks}
    number_of_transfers = {w: lpSum(transfer_out[p][w] for p in players) for w in gameweeks}
    number_of_transfers[next_gw-1] = 1
    transfer_diff = {w: number_of_transfers[w] - free_transfers[w] for w in gameweeks}
    # initial constraints
    for p in players:
      if p in initial_squad:
        prob += squad[p][next_gw-1] == 1
      else:
        prob += squad[p][next_gw-1] == 0
    prob += in_the_bank[next_gw-1] == itb
    prob += free_transfers[next_gw-1] == ft
    # set constraints

    for w in gameweeks:
        prob += squad_count[w] == 15
        prob += captain_count[w] == 1
        prob += vicecap_count[w] == 1
        prob += lpSum(lineup[p][w] for p in players) == 11
        prob += lpSum(bench[p][w][0] for p in players if player_type[p] == 1) == 1

        # free transfer constraints
        prob += free_transfers[w] == aux[w] + 1
        prob += free_transfers[w-1] - number_of_transfers[w] <= 2 * aux[w]
        prob += free_transfers[w-1] - number_of_transfers[w] >= aux[w] + (-14)*(1 - aux[w])
        prob += penalized_transfers[w] >= transfer_diff[w]
        prob += in_the_bank[w] == in_the_bank[w-1] + sold_amount[w] - bought_amount[w]

        for o in [1, 2, 3]:
          prob += lpSum(bench[p][w][o] for p in players) == 1
        for p in players:
            prob += squad[p][w] >= lineup[p][w]
            prob += lineup[p][w] >= captain[p][w]
            prob += lineup[p][w] >= vicecap[p][w]
            prob += captain[p][w] + vicecap[p][w] <= 1
            # transfer constraints
            prob += squad[p][w] == squad[p][w-1] + transfer_in[p][w] - transfer_out[p][w]
            prob += lpSum(bench[p][w][o] for o in order) + lineup[p][w] <= 1
            for o in order:
              prob += squad[p][w] >= bench[p][w][o]
        for t in element_types:
            prob += lineup_type_count[t, w] <= type_data.loc[t, 'squad_max_play']
            prob += lineup_type_count[t, w] >= type_data.loc[t, 'squad_min_play']
            prob += squad_type_count[t, w] == type_data.loc[t, 'squad_select']
        for t in teams:
            prob += lpSum(squad[p][w] for p in players if merged_data.loc[p, 'Team'] == t) <= 3
    # set objective
    gw_xp = {w: lpSum(points_player_week[p, w] * (lineup[p][w] + captain[p][w] + 0.1* vicecap[p][w] + lpSum(bench_weights[o] * bench[p][w][o] for o in order)) for p in players) for w in gameweeks}
    gw_total = {w: gw_xp[w] - 4 * penalized_transfers[w] for w in gameweeks}
    if objective == 'regular':
        total_xp = lpSum(gw_total[w] for w in gameweeks)
        prob += total_xp
    else:
        decay_objective = lpSum(gw_total[w] * pow(decay_base, w-next_gw) for w in gameweeks)
        prob += decay_objective
    # solve
    prob.writeLP(f'optimizers/{problem_name}.lp');
    prob.solve(PULP_CBC_CMD(msg=0));
    picks = []
    for w in gameweeks:
        for p in players:
            is_cap = 0
            is_vice = 0
            is_lineup = 0
            is_transfer_in = 0
            is_transfer_out = 0
            if squad[p][w].varValue + transfer_out[p][w].varValue == 1:
                if captain[p][w].varValue == 1:
                    is_cap = 1
                if vicecap[p][w].varValue == 1:
                    is_vice = 1
                if lineup[p][w].varValue == 1:
                    is_lineup = 1
                if transfer_in[p][w].varValue == 1:
                    is_transfer_in = 1
                if transfer_out[p][w].varValue == 1:
                    is_transfer_out = 1
                bench_value = -1
                for o in order:
                  if bench[p][w][o].varValue == 1:
                    bench_value = o
                picks.append([
                    w,
                    type_data.loc[merged_data.loc[p, 'element_type'], 'singular_name_short'],
                    merged_data.loc[p, 'element_type'],
                    merged_data.loc[p, 'web_name'],
                    merged_data.loc[p, 'name'],
                    player_price[p],
                    points_player_week[p, w],
                    is_lineup,
                    bench_value,
                    is_cap,
                    is_vice,
                    is_transfer_in,
                    is_transfer_out
                ])
    picks_df = pd.DataFrame(picks, columns = [
        'week',
        'pos',
        'element_type',
        'name',
        'team',
        'price',
        'xPts',
        'is_lineup',
        'bench',
        'is_cap',
        'is_vice',
        'is_transfer_in',
        'is_transfer_out'
    ]).sort_values(by=[
        'week',
        'is_lineup',
        'element_type',
        'xPts'
    ], ascending=[
        True,
        False,
        True,
        True
    ])
    total_points = value(lpSum((lineup[p][w] + captain[p][w]) * points_player_week[p, w] for p in players for w in gameweeks))
    summary_of_actions = ""
    for w in gameweeks:
        summary_of_actions += f"** GW {w}:\n"
        summary_of_actions += f"ITB - {in_the_bank[w].varValue}, FT - {free_transfers[w].varValue}, PT - {penalized_transfers[w].varValue} , xPts - {round(value(lpSum((lineup[p][w] + captain[p][w]) * points_player_week[p, w] for p in players)), 2)}\n"
        for p in players:
            if transfer_in[p][w].varValue == 1:
                summary_of_actions += f"Buy {p} - {merged_data['web_name'][p]}\n"
            if transfer_out[p][w].varValue == 1:
                summary_of_actions += f"Sell {p} - {merged_data['web_name'][p]}\n"
    return {
        'picks': picks_df,
        'total_xPts': total_points,
        'summary': summary_of_actions
    }

sol = multi_period_solver(
    team_id=151180,
    gw=26,
    horizon = 5,
    objective = 'decay',
    team_override = False,
    team = []
)
print(sol['summary'])
