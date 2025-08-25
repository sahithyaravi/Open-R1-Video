def get_data(item):
    # dataset = wangxd
    video_path = item.get("video") or item.get("video_path")

    if "wangxd" in video_path:
        ground_truth = item.get("ground_truth")

    else:
        ground_truth = item.get("caption")
        video_path = "data/ActivityNet_Captions/Activity_Videos/" + video_path

    return video_path, ground_truth
