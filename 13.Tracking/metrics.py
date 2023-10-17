
def iou_score(bbox1, bbox2):
    """Jaccard index or Intersection over Union.

    https://en.wikipedia.org/wiki/Jaccard_index

    bbox: [xmin, ymin, xmax, ymax]
    """
    assert len(bbox1) == 4
    assert len(bbox2) == 4

    x_low = max(bbox1[0], bbox2[0])
    y_low = max(bbox1[1], bbox2[1])

    x_up = min(bbox1[2], bbox2[2])
    y_up = min(bbox1[3], bbox2[3])

    intersection = max(0, x_up - x_low) * max(0, y_up - y_low)

    S_bbox1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    S_bbox2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    union = S_bbox1 + S_bbox2 - intersection

    return float(intersection) / union


def motp(obj, hyp, threshold=0.5):
    """Calculate MOTP

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0

    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Write code here

        # Step 1: Convert frame detections to dict with IDs as keys
        n = len(frame_obj)
        m = len(frame_hyp)
        obj_dict = {}
        hyp_dict = {}
        for i in range(n):
            obj_dict[frame_obj[i][0]] = frame_obj[i][1:]
        for j in range(m):
            hyp_dict[frame_hyp[j][0]] = frame_hyp[j][1:]

        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        Is = []
        Js = []
        for match in matches:
            for i in hyp_dict:
                for j in obj_dict:
                    if j == match and i == matches[j]:
                        iou = iou_score(hyp_dict[i], obj_dict[j])
                        if iou > threshold:
                            dist_sum += iou
                            match_count += 1
                            Is.append(i)
                            Js.append(j)
        for i in Is:
            del hyp_dict[i]

        for j in Js:
            del obj_dict[j]

        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold
        ious = []
        for i in hyp_dict:
            for j in obj_dict:
                iou = iou_score(hyp_dict[i], obj_dict[j])
                if iou > threshold:
                    ious.append((iou, j, i))

        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        ious = sorted(ious)
        for iou, i, j in ious:
            if not i in matches:
                matches[i] = j
                dist_sum += iou
                match_count += 1
                del hyp_dict[j]
                del obj_dict[i]

    # Step 6: Calculate MOTP
    MOTP = dist_sum / match_count

    return MOTP


def motp_mota(obj, hyp, threshold=0.5):
    """Calculate MOTP/MOTA

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0
    missed_count = 0
    false_positive = 0
    mismatch_error = 0
    g = 0

    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Step 1: Convert frame detections to dict with IDs as keys
        n = len(frame_obj)
        g += n
        m = len(frame_hyp)
        obj_dict = {}
        hyp_dict = {}
        for i in range(n):
            obj_dict[frame_obj[i][0]] = frame_obj[i][1:]
        for j in range(m):
            hyp_dict[frame_hyp[j][0]] = frame_hyp[j][1:]
        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete m atched detections from frame detections
        Is = []
        Js = []
        for match in matches:
            for i in hyp_dict:
                for j in obj_dict:
                    if j == match and i == matches[j]:
                        iou = iou_score(hyp_dict[i], obj_dict[j])
                        if iou > threshold:
                            dist_sum += iou
                            match_count += 1
                            Is.append(i)
                            Js.append(j)
        print("1")
        for i in Is:
            del hyp_dict[i]

        for j in Js:
            del obj_dict[j]
        print('HERE')
        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold
        ious = []
        for i in hyp_dict:
            for j in obj_dict:
                iou = iou_score(hyp_dict[i], obj_dict[j])
                if iou > threshold:
                    ious.append((iou, j, i))
        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        ious = sorted(ious)
        for iou, i, j in ious:
            if not i in matches:
                matches[i] = j
                dist_sum += iou
                match_count += 1
                del hyp_dict[j]
                del obj_dict[i]
            elif i in matches and matches[i] != j:
                mismatch_error += 1
                matches[i] = j
                del hyp_dict[j]
                del obj_dict[i]

        # Step 5: If matched IDs contradict previous matched IDs - increase mismatch error

        # Step 6: Update matches with current matched IDs

        # Step 7: Errors
        # All remaining hypotheses are considered false positives
        false_positive += len(hyp_dict)
        # All remaining objects are considered misses
        missed_count += len(obj_dict)

    # Step 8: Calculate MOTP and MOTA
    MOTP = float(dist_sum) / match_count
    MOTA = 1 - float(missed_count + false_positive + mismatch_error) / g

    return MOTP, MOTA
