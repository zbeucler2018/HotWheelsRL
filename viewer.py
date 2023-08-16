import gymnasium as gym
import pygame
import numpy as np

class Viewer(gym.Wrapper):
    """
    Wrapper that shows the game frame and the
    CNN input.
    NOTE: Needs to be the last wrapper applied
    NOTE: Works with single env (not sure with vec envs)
    """

    def __init__(self, env):
        super().__init__(env)

        pygame.init()
        self.fpsClock = pygame.time.Clock()
        self.WINDOW_WIDTH = 800
        self.WINDOW_HEIGHT = 400

        self.WINDOW = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        self.WINDOW.fill((0, 0, 0))

        self.font = pygame.font.SysFont(None, 30)

        pygame.display.set_caption("HotWheelsRL Viewer")

    def update_display(self, obs, em_img):
        """
        Update the display with new em_img and obs
        """

        self.em_surface = pygame.surfarray.make_surface(np.transpose(em_img, (1, 0, 2)))
        self.WINDOW.blit(self.em_surface, (0, 0))
        if obs.shape[-1] == 1:  # obs is grayscale
            self.obs_surface = pygame.transform.flip(
                pygame.transform.rotate(
                    pygame.surfarray.make_surface(obs[:, :, 0]), 270.0
                ),
                flip_y=False,
                flip_x=True,
            )
            self.obs_surface.set_palette([(i, i, i) for i in range(256)])
        else:
            self.obs_surface = pygame.transform.flip(
                pygame.transform.rotate(pygame.surfarray.make_surface(obs), 270.0),
                flip_y=False,
                flip_x=True,
            )
        self.WINDOW.blit(self.obs_surface, (400, 0))
        # render text
        self.em_text = self.font.render(
            f"Emulator Frame - {em_img.shape}", False, (255, 255, 255)
        )
        self.WINDOW.blit(self.em_text, (50, 200))

        self.obs_text = self.font.render(
            f"CNN Input - {obs.shape}", False, (255, 255, 255)
        )
        self.WINDOW.blit(self.obs_text, (550, 200))

        pygame.display.flip()
        pygame.display.update()

    def close(self, **kwargs):
        pygame.quit()
        return super().close(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        em_img = self.env.get_screen()
        self.update_display(observation, em_img)
        self.fpsClock.tick()
        return observation, reward, terminated, truncated, info