import random
import matplotlib.pyplot as plt

matrix = [[0.33, 0.33, 0.34],
          [0.34, 0.33, 0.34],
          [0.33, 0.34, 0.33]]

def choose_move(state, player_move_history):
    moves = ["kamień", "papier", "nożyce"]
    weights = matrix[state]

    if player_move_history and player_move_history[-1] == "kamień":
        weights = [weights[0], weights[1] + 0.1, weights[2]]

    elif player_move_history and player_move_history[-1] == "nożyce":
        weights = [weights[0] + 0.1, weights[1], weights[2]]

    elif player_move_history and player_move_history[-1] == "papier":
        weights = [weights[0], weights[1], weights[2] + 0.1]

    return random.choices(moves, weights=weights)[0]

def compare_moves(player_move, computer_move):
    if player_move == computer_move:
        return 0
    elif player_move == "kamień" and computer_move == "nożyce" or \
         player_move == "papier" and computer_move == "kamień" or \
         player_move == "nożyce" and computer_move == "papier":
        return 1
    else:
        return -1

def play_games(num_games):
    player_money = 0
    money_history = [player_money]
    state = 0
    player_move_history = []
    for i in range(num_games):
        player_move = input("\nWybierz ruch (kamień, papier, nożyce), lub naciśnij 'q' by zakończyć grę: ")
        if player_move == 'q':
            break
        player_move_history.append(player_move)
        computer_move = choose_move(state, player_move_history)
        result = compare_moves(player_move, computer_move)
        player_money += result
        money_history.append(player_money)
        print(f"\nGracz: {player_move}, Komputer: {computer_move}, Wynik: {result}, Kasa: {player_money}\n")
        if player_move == "kamień":
             state = (state + 1) % 3
        elif player_move == "nożyce":
             state = (state - 1) % 3
        elif player_move == "papier":
            state = state
    return money_history


def diagram(money_history):
    plt.plot(money_history)
    plt.xlabel('Liczba gier')
    plt.ylabel('Kasa')
    plt.title('Zmiana stanu kasy w każdym kroku gry')
    plt.show()

def main():
    num_games = 10
    money_history = play_games(num_games)
    diagram(money_history)

if __name__ == '__main__':
    main()
