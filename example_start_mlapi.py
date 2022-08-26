"""A Script to start the MLAPI server."""


if __name__ == "__main__":
    from argparse import ArgumentParser
    from zm_mlapi.app import MLAPI

    parser = ArgumentParser()
    parser.add_argument("-C", "--config", help="Path to configuration ENV file", default="./prod.env")
    args = vars(parser.parse_args())

    # setup directories and confirm settings

    # Starts the FastAPI server
    server: MLAPI = MLAPI(args["config"])



