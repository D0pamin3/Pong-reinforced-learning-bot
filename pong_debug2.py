input("Press ENTER to launch Pong...")


import pygame
import sys
import random
import time
import csv
import numpy as np


MODE = input("Arduino Mode? (YES / NO): ").strip().upper()
ARDUINO_MODE = MODE == "YES"


if ARDUINO_MODE:
    import serial
    try:
        ser = serial.Serial("COM7", 9600, timeout=0)
        print("Arduino connected!")
    except:
        print("Arduino not found. Exiting.")
        sys.exit()


pygame.init()
WIDTH, HEIGHT = 1000, 640
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Advanced RL Pong Research Environment")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 18)


WHITE = (240, 240, 240)
GREEN = (0, 220, 0)
RED = (220, 0, 0)
BLACK = (0, 0, 0)
GRAY = (40, 40, 40)


PADDLE_W = 14
PADDLE_H = 120
BALL_SIZE = 14

left_paddle = pygame.Rect(40, HEIGHT // 2 - PADDLE_H // 2, PADDLE_W, PADDLE_H)
right_paddle = pygame.Rect(WIDTH - 54, HEIGHT // 2 - PADDLE_H // 2, PADDLE_W, PADDLE_H)
ball = pygame.Rect(WIDTH // 2, HEIGHT // 2, BALL_SIZE, BALL_SIZE)

BALL_SPEED = 3.0  # Slower and stable
ball_dx = random.choice([-1, 1])
ball_dy = random.choice([-1, 1])

PADDLE_SPEED = 6


ACTIONS = [-1, 0, 1]  # up, stay, down

Q1 = {}
Q2 = {}
alpha = 0.08
gamma = 0.99
lambda_ = 0.85

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995

reward_total = 0.0
hits = 0
misses = 0
idle_steps = 0
steps = 0


def discretize(val, max_val, bins):
    return min(bins - 1, max(0, int((val / max_val) * bins)))

def get_state():
    ball_y = discretize(ball.centery, HEIGHT, 18)
    paddle_y = discretize(right_paddle.centery, HEIGHT, 18)
    dy = 1 if ball_dy > 0 else 0
    dx = 1 if ball_dx > 0 else 0
    speed_bin = discretize(BALL_SPEED, 5.0, 5)
    return (ball_y, paddle_y, dy, dx, speed_bin)


def choose_action(state):
    if random.random() < epsilon or state not in Q1:
        return random.choice(ACTIONS)
    qsum = {a: Q1[state].get(a, 0) + Q2[state].get(a, 0) for a in ACTIONS}
    return max(qsum, key=qsum.get)

def update_q(state, action, reward, next_state):
    Q1.setdefault(state, {a: 0.0 for a in ACTIONS})
    Q2.setdefault(state, {a: 0.0 for a in ACTIONS})
    Q1.setdefault(next_state, {a: 0.0 for a in ACTIONS})
    Q2.setdefault(next_state, {a: 0.0 for a in ACTIONS})
    if random.random() < 0.5:
        best_next = max(Q1[next_state], key=Q1[next_state].get)
        td_error = reward + gamma * Q2[next_state][best_next] - Q1[state][action]
        Q1[state][action] += alpha * td_error
    else:
        best_next = max(Q2[next_state], key=Q2[next_state].get)
        td_error = reward + gamma * Q1[next_state][best_next] - Q2[state][action]
        Q2[state][action] += alpha * td_error


start_time = time.time()
csv_file = "pong_ai_dqn.csv"
csv_writer = open(csv_file, "w", newline="")
writer = csv.writer(csv_writer)
writer.writerow(["MINUTE", "REWARD_TOTAL", "HITS", "MISSES", "EPSILON", "SIGNAL", "RIGHT_Y"])

last_state = None
last_action = None
running = True

while running:
    clock.tick(60)
    steps += 1

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if ARDUINO_MODE:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] and left_paddle.top > 0:
            left_paddle.y -= PADDLE_SPEED
        if keys[pygame.K_s] and left_paddle.bottom < HEIGHT:
            left_paddle.y += PADDLE_SPEED
    else:
        if left_paddle.centery < ball.centery:
            left_paddle.y += PADDLE_SPEED
        elif left_paddle.centery > ball.centery:
            left_paddle.y -= PADDLE_SPEED
        left_paddle.y = max(0, min(HEIGHT-PADDLE_H, left_paddle.y))

    state = get_state()
    action = choose_action(state)
    right_paddle.y += action * PADDLE_SPEED
    right_paddle.y = max(0, min(HEIGHT - PADDLE_H, right_paddle.y))

    ball.x += ball_dx * BALL_SPEED
    ball.y += ball_dy * BALL_SPEED

    # Wall bounce
    if ball.top <= 0:
        ball.top = 0
        ball_dy *= -1
    if ball.bottom >= HEIGHT:
        ball.bottom = HEIGHT
        ball_dy *= -1

    reward = 0.0
    signal = "NONE"

    if ball.colliderect(right_paddle):
        ball_dx *= -1
        reward += 1
        hits += 1

    if ball.colliderect(left_paddle):
        ball_dx *= -1

    if ball.right >= WIDTH:
        reward -= 1
        misses += 1
        ball.center = (WIDTH // 2, HEIGHT // 2)
        ball_dx = -1
        ball_dy = random.choice([-1, 1])

    if ball.left <= 0:
        reward += 1
        hits += 1
        ball.center = (WIDTH // 2, HEIGHT // 2)
        ball_dx = 1
        ball_dy = random.choice([-1, 1])

    if ARDUINO_MODE:
        try:
            data = ser.readline().decode().strip()
            if data == "YES":
                reward = 1
                signal = "YES"
            elif data == "NO":
                reward = -1
                signal = "NO"
        except:
            pass

    if last_state is not None:
        update_q(last_state, last_action, reward, state)
    last_state = state
    last_action = action
    reward_total += reward
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    minute = int((time.time() - start_time)//60)
    writer.writerow([minute, reward_total, hits, misses, round(epsilon,3), signal, right_paddle.y])
    csv_writer.flush()

    screen.fill(BLACK)
    pygame.draw.rect(screen, WHITE, left_paddle)
    pygame.draw.rect(screen, WHITE, right_paddle)
    pygame.draw.ellipse(screen, WHITE, ball)

    hud = [
        f"MODE: {'ARDUINO' if ARDUINO_MODE else 'AUTO'}",
        f"Reward Total: {round(reward_total,2)}",
        f"Hits: {hits}",
        f"Misses: {misses}",
        f"Epsilon: {round(epsilon,3)}",
        f"Ball Speed: {BALL_SPEED}",
        f"Steps: {steps}",
        f"State: {state}",
        f"Action: {action}"
    ]

    for i, text in enumerate(hud):
        screen.blit(font.render(text, True, GREEN), (20, 20 + i*22))

    pygame.display.flip()

csv_writer.close() #final i think
pygame.quit()
sys.exit()

