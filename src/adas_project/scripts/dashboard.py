import pygame
import json
import sys
import time

def run_dashboard():
    pygame.init()
    WIDTH, HEIGHT = 800, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("ADAS Research Live Dashboard")
    
    font = pygame.font.SysFont("monospace", 18)
    title_font = pygame.font.SysFont("monospace", 28, bold=True)
    
    clock = pygame.time.Clock()
    
    path = '../results/live_stats.json'
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        try:
            with open(path, 'r') as f:
                stats = json.load(f)
        except:
            stats = {"driver": "WAITING", "level": 0, "system": "NONE", 
                     "collisions": 0, "offroads": 0, "total": 0, "current_episode": 0,
                     "current_phase": "INITIALIZING..."}
                     
        screen.fill((20, 25, 30))
        
        # Title
        t = title_font.render(f"ADAS PERFORMANCE METRICS", True, (255, 255, 255))
        screen.blit(t, (20, 20))
        
        y = 80
        color = (150, 200, 255)
        
        lines = [
            f"Phase:   {stats.get('current_phase', 'RUNNING')}",
            f"Driver:  {stats.get('driver', '')}",
            f"Level:   {stats.get('level', 0)}",
            f"System:  {stats.get('system', '')}",
            f"Episode: {stats.get('current_episode', 0)} / 200",
            "",
            "--- L2+ SAFETY CONSTRAINTS ---",
            f"Collisions: {stats.get('collisions', 0)}",
            f"Offroads:   {stats.get('offroads', 0)}",
            "",
            "--- TELEMETRY ---",
            f"Last Min TTC: {stats.get('last_min_ttc', 0.0):.2f}s",
            f"Total Episodes Run: {stats.get('total', 0)} / 6000"
        ]
        
        for line in lines:
            if "Collisions" in line and stats.get('collisions', 0) > 0:
                c = (255, 50, 50)
            elif "Offroads" in line and stats.get('offroads', 0) > 0:
                c = (255, 50, 50)
            elif "SAFETY" in line:
                c = (255, 255, 100)
            else:
                c = color
            
            s = font.render(line, True, c)
            screen.blit(s, (30, y))
            y += 35
            
        pygame.display.flip()
        clock.tick(2) # 2 Hz refresh
        
    pygame.quit()

if __name__ == '__main__':
    run_dashboard()
