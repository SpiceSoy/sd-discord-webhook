import launch

# TODO: add pip dependency if need extra module only on extension

if not launch.is_installed("discord-webhook"):
    launch.run_pip("install discord-webhook==1.1.0", "requirements for sd-discord-webhook")
    launch.run_pip("install uvicorn", "requirements for sd-discord-webhook")
    launch.run_pip("install fastapi", "requirements for sd-discord-webhook")
    launch.run_pip("install diffusers", "requirements for sd-discord-webhook")
