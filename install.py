import launch

if not launch.is_installed("lavis"):
    try:
        launch.run_pip("install salesforce-lavis", "requirements for lavis")
    except:
        print(
            "Can't install salesforce-lavis. Please follow the readme to install manually"
        )
