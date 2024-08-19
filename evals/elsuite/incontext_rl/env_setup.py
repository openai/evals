"""
Optional setup scripts for specific environments.
"""

def setup_GymnasiumBandits():
    import gymnasium_bandits
    return

ENV_SETUP_FUNCS = {
    "BanditTwoArmedHighLowFixed-v0": setup_GymnasiumBandits,
    "BanditTenArmedRandomFixed-v0": setup_GymnasiumBandits,
}