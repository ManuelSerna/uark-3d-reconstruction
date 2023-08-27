import argparse


def get_base_parser():
    parser = argparse.ArgumentParser("Args Parser", add_help=False)
    
    # Cmd parameters
    parser.add_argument("--depth", type=str, required=True, help="depth image with pixel values in the range 0..255")
    
    # Camera settings
    parser.add_argument("--focal", type=float, default=721.5377, help="focal length for the camera intrinsics")
    
    return parser
