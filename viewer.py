
import pygame as pg
 
from gymnasium import Wrapper
from gymnasium.core import Env






class Viewer(Wrapper):

    def __init__(self, env: Env):
        super().__init__(env)

        pg.init()
        self.screen = pg.display.set_mode((args.display_width, args.display_height))
        self.surface = pg.Surface((FB_WIDTH, FB_HEIGHT))
        self.surface.set_colorkey((0,0,0))

    def step(self, action):
        pass

    def close(self):
        pg.quit()
        return super().close()















# Colours
# BACKGROUND = (255, 255, 255)
 
# # Game Setup
# FPS = 60
# fpsClock = pygame.time.Clock()
# WINDOW_WIDTH = 400
# WINDOW_HEIGHT = 300
 
# WINDOW = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
# pygame.display.set_caption('My Game!')
 
# # The main function that controls the game
# def main () :
#   looping = True
  
#   # The main game loop
#   while looping :
#     # Get inputs
#     for event in pygame.event.get():
#       if event.type == QUIT:
#         pygame.quit()
#         sys.exit()
    
#     # Processing
#     # This section will be built out later
 
#     # Render elements of the game
#     WINDOW.fill(BACKGROUND)
#     pygame.display.update()
#     fpsClock.tick(FPS)
 
# main()




