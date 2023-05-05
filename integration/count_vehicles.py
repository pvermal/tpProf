from count_vehicles_images import count_vehicles_images as imagesStream
from count_vehicles_webcam import count_vehicles_webcam as webcamStream
import argparse
import time


def parse_args():
    # initialize the parser
    parser = argparse.ArgumentParser(description="Count Vehicles")

    # add arguments
    parser.add_argument(
        "--stream",
        help="Stream to take the data from. webcam / images.",
        type=str,
        default="webcam",
    )
    parser.add_argument(
        "--images_dir",
        help="Directory to the images. Only for 'images' stream.",
        type=str,
        default="./videos/5fps/puente_BA_centro_lejos_1_5fps/images",
    )
    parser.add_argument("--display", help="Display results.", type=bool, default=False)
    parser.add_argument(
        "--plot_dets", help="Plot detections.", type=bool, default=False
    )
    parser.add_argument(
        "--plot_tracks",
        help="Plot tracking algorithm estimations.",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--save_video", help="Save video with results.", type=bool, default=False
    )
    parser.add_argument(
        "--output_video_path", help="Specify output video path.", type=str, default="./videos/out_{}.mp4".format(time.time())
    )
    parser.add_argument(
        "--showLogTimes",
        help="Show processing times for each stage.",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--is_yolov8",
        help="Choose model. By default it will use yolov5",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--iou_threshold_tracking",
        help="Specify IoU threshold for the tracking algorithm.",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--max_age",
        help="Maximum number of frames to keep alive a track without associated detections in the tracking algorithm.",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--min_hits",
        help="Minimum number of associated detections before track is initialised in the tracking algorithm.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--number_of_lanes", help="Number of lanes to measure.", type=int, default=-1
    )
    parser.add_argument(
        "--fps", help="Aproximate fps of the selected stream.", type=int, default=10
    )

    parser.add_argument(
        "--show_plot_live", help="Show in real time (only buffer delay) the output of each lane. Notice that this feature makes the fps to drop significantly.", type=bool, default=False
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    if args.stream == "webcam":
        # example:
        # python ./integration/count_vehicles.py --stream webcam --display True --plot_dets True --plot_tracks True
        webcamStream(
            display=args.display,
            plot_dets=args.plot_dets,
            plot_tracks=args.plot_tracks,
            save_video=args.save_video,
            output_video_path=args.output_video_path,
            showLogTimes=args.showLogTimes,
            is_yolov8=args.is_yolov8,
            iou_threshold_tracking=args.iou_threshold_tracking,
            max_age=args.max_age,
            min_hits=args.min_hits,
            number_of_lanes=args.number_of_lanes,
            fps=args.fps,
            show_plot_live=args.show_plot_live
        )
    # ! Some functionalities available in "webcam" stream may not be implemented in "images" stream yet
    elif args.stream == "images":
        # example
        # python python ./integration/count_vehicles.py --stream images --display True --plot_dets True --plot_tracks True
        imagesStream(
            display=args.display,
            images_dir=args.images_dir,
            plot_dets=args.plot_dets,
            plot_tracks=args.plot_tracks,
            save_video=args.save_video,
            output_video_path=args.output_video_path,
            showLogTimes=args.showLogTimes,
            is_yolov8=args.is_yolov8,
            iou_threshold_tracking=args.iou_threshold_tracking,
            max_age=args.max_age,
            min_hits=args.min_hits,
            number_of_lanes=args.number_of_lanes,
            fps=args.fps,
        )
    else:
        print("Error. Invalid Stream.")
