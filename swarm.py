import pygame
import numpy as np
import heapq

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Swarm Robot Navigation with A* Algorithm")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GRAY = (169, 169, 169)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
ORANGE = (255, 165, 0)

# Robot class
class Robot:
    def __init__(self, x, y, radius=10, sight_range=100):
        self.x = x
        self.y = y
        self.radius = radius
        self.sight_range = sight_range
        self.color = BLUE
        self.speed = 2
        self.reached_goal = False

    def draw(self, screen, obstacles):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
        self.draw_lidar(screen, obstacles)

    def draw_lidar(self, screen, obstacles):
        num_rays = 36
        angle_step = 360 / num_rays
        for i in range(num_rays):
            angle = np.radians(i * angle_step)
            end_x = self.x + self.sight_range * np.cos(angle)
            end_y = self.y + self.sight_range * np.sin(angle)
            color = GREEN
            for obstacle in obstacles:
                if obstacle.clipline(self.x, self.y, end_x, end_y):
                    color = RED
                    break
                elif np.linalg.norm(np.array([self.x, self.y]) - np.array([end_x, end_y])) < self.sight_range / 2:
                    color = ORANGE
            pygame.draw.line(screen, color, (self.x, self.y), (end_x, end_y), 1)

    def move_towards(self, path, robots):
        if path and not self.reached_goal:
            next_pos = path[0]
            if not any(np.linalg.norm(np.array([next_pos[0], next_pos[1]]) - np.array([robot.x, robot.y])) < self.radius * 2 for robot in robots if robot != self):
                self.x, self.y = path.pop(0)
            if len(path) == 0:
                self.reached_goal = True

# Create robots
def create_robots(num_robots, sight_range):
    robots = []
    for _ in range(num_robots):
        x = np.random.randint(50, WIDTH - 50)
        y = np.random.randint(50, HEIGHT - 50)
        robots.append(Robot(x, y, sight_range=sight_range))
    return robots

# Create obstacles
def create_obstacles(num_obstacles):
    obstacles = []
    for _ in range(num_obstacles):
        x = np.random.randint(50, WIDTH - 50)
        y = np.random.randint(50, HEIGHT - 50)
        width = np.random.randint(20, 50)
        height = np.random.randint(20, 50)
        obstacles.append(pygame.Rect(x, y, width, height))
    return obstacles

# Check for collision with obstacles
def is_collision(x, y, obstacles):
    for obstacle in obstacles:
        if obstacle.collidepoint(x, y):
            return True
    return False

# A* algorithm for pathfinding
def a_star(start, goal, obstacles):
    def heuristic(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        neighbors = [(current[0] + dx, current[1] + dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
        for neighbor in neighbors:
            if 0 <= neighbor[0] < WIDTH and 0 <= neighbor[1] < HEIGHT and not is_collision(neighbor[0], neighbor[1], obstacles):
                tentative_g_score = g_score[current] + heuristic(current, neighbor)
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return []

# Main function
def main():
    num_robots = 3
    sight_range = 100
    robots = create_robots(num_robots, sight_range)
    target = (WIDTH // 2, HEIGHT // 2)
    obstacles = create_obstacles(5)
    paths = [[] for _ in range(num_robots)]
    running = True
    status = "Waiting for target selection"

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                target = event.pos
                status = "Path calculating"
                paths = [a_star((robot.x, robot.y), target, obstacles) for robot in robots]
                status = "Calibration"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_o:
                    obstacles.append(pygame.Rect(np.random.randint(50, WIDTH - 50), np.random.randint(50, HEIGHT - 50), np.random.randint(20, 50), np.random.randint(20, 50)))
                if event.key == pygame.K_UP:
                    num_robots += 1
                    robots = create_robots(num_robots, sight_range)
                    paths = [[] for _ in range(num_robots)]
                if event.key == pygame.K_DOWN:
                    num_robots = max(1, num_robots - 1)
                    robots = create_robots(num_robots, sight_range)
                    paths = [[] for _ in range(num_robots)]

        screen.fill(WHITE)

        for obstacle in obstacles:
            pygame.draw.rect(screen, GRAY, obstacle)

        all_reached_goal = True
        for i, robot in enumerate(robots):
            if not robot.reached_goal:
                all_reached_goal = False
            robot.move_towards(paths[i], robots)
            robot.draw(screen, obstacles)

        # Check if all robots are near the goal
        all_near_goal = all(np.linalg.norm(np.array([robot.x, robot.y]) - np.array(target)) < robot.sight_range for robot in robots)

        if all_near_goal:
            status = "Goal reached successfully"

        pygame.draw.circle(screen, RED, target, 5)

        font = pygame.font.Font(None, 36)
        text_surface = font.render(status, True, BLACK)
        screen.blit(text_surface, (10, 10))

        pygame.display.flip()
        pygame.time.delay(30)

    pygame.quit()

if __name__ == "__main__":
    main()
