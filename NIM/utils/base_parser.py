import argparse


def get_base_parser():
    parser = argparse.ArgumentParser("Args Parser", add_help=False)
    
    # Cmd parameters
    parser.add_argument(
        "--depth",
        type=str,
        default=None,
        help="depth image with pixel values in the range 0..255, exclusive with --folder"
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="folder containing depth images, exclusive with --depth"
    )
    parser.add_argument(
        "--outpath",
        type=str,
        default=None,
        help="output folder path"
    )

    # Camera settings
    parser.add_argument("--focal", type=float, default=721.5377, help="focal length for the camera intrinsics")
    
    return parser
