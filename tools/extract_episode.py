import os
import io
import json
import tarfile
import argparse

import pandas as pd


def pose_to_position_rotation(T):
    """
    4x4 pose matrix

    return:
        position [x,y,z]
        rotation 3x3
    """

    position = [
        float(T[0][3]),
        float(T[1][3]),
        float(T[2][3]),
    ]

    rotation = [
        [float(v) for v in row[:3]]
        for row in T[:3]
    ]

    return position, rotation


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--tar", required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    rgb_dir = os.path.join(args.output, "rgb")
    depth_dir = os.path.join(args.output, "depth")

    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    episode_name = f"episode_{args.episode:06d}"

    print("opening tar...")
    tar = tarfile.open(args.tar, "r:gz")

    print("loading parquet...")

    parquet_path = (
        f"D7N2EKCX4Sj/data/chunk-000/"
        f"{episode_name}.parquet"
    )

    parquet_file = tar.extractfile(parquet_path)

    df = pd.read_parquet(parquet_file)

    print("frames =", len(df))

    trajectory = []

    for _, row in df.iterrows():

        frame_id = int(row["frame_index"])

        rgb_tar_path = (
            "D7N2EKCX4Sj/videos/chunk-000/"
            "observation.images.rgb.125cm_0deg/"
            f"{episode_name}_{frame_id}.jpg"
        )

        depth_tar_path = (
            "D7N2EKCX4Sj/videos/chunk-000/"
            "observation.images.depth.125cm_0deg/"
            f"{episode_name}_{frame_id}.png"
        )

        rgb_name = f"{frame_id:06d}.jpg"
        depth_name = f"{frame_id:06d}.png"

        rgb_out = os.path.join(rgb_dir, rgb_name)
        depth_out = os.path.join(depth_dir, depth_name)

        rgb_file = tar.extractfile(rgb_tar_path)

        with open(rgb_out, "wb") as f:
            f.write(rgb_file.read())

        depth_file = tar.extractfile(depth_tar_path)

        with open(depth_out, "wb") as f:
            f.write(depth_file.read())

        pose = row["pose.125cm_0deg"]

        position, rotation = pose_to_position_rotation(pose)

        trajectory.append(
            {
                "frame_id": frame_id,
                "rgb": f"rgb/{rgb_name}",
                "depth": f"depth/{depth_name}",
                "position": position,
                "rotation": rotation,
            }
        )

    json_path = os.path.join(
        args.output,
        "trajectory.json"
    )

    with open(json_path, "w") as f:
        json.dump(
            trajectory,
            f,
            indent=2
        )

    print()
    print("done")
    print("trajectory =", json_path)


if __name__ == "__main__":
    main()