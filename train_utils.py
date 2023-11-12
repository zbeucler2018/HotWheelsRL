from utils import HotWheelsStates
import retro



def make_retro(
    *,
    game,
    state: HotWheelsStates = HotWheelsStates.DEFAULT,
    render_mode="rgb_array",
    **kwargs,
):
    env = retro.make(
        game,
        state=f"{state}.state",
        info=retro.data.get_file_path(
            "HotWheelsStuntTrackChallenge-gba", f"{state}.json"
        ),
        render_mode=render_mode,
        **kwargs,
    )

    return env
