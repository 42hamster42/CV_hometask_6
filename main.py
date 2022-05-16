import cv2
import os
import numpy as np


def binarize(im):
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # th, res = cv2.threshold(im_gray, 128, 255, cv2.THRESH_OTSU)
    res = cv2.adaptiveThreshold(im_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 199, 5)
    # res = cv2.threshold(im_gray, int(th), 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel, iterations=2)
    # print(f'Otsu treshold {th}')
    return res


def to_rle(im):
    result = []
    for y, line in enumerate(im):
        # white = True
        last_change_idx = 0
        last_value = 0
        line_res = []
        for x, p in enumerate(line):
            if p != last_value:
                line_res.append((last_change_idx, x))
                last_change_idx = x
                last_value = p
        result.append(line_res)
    return result


def heat_map_likelihood(size, rle, threshold):
    modules_parts = [1, 1, 3, 1, 1]
    result = np.zeros(size, np.uint8)
    assert size[0] == len(rle)
    all_modules_widths = []
    for y, line in enumerate(rle):
        for istart in range(0, len(line) - 4, 2):
            ilast = istart + 4
            iend = istart + 5
            assert ilast < len(line)
            assert iend <= len(line)
            range_start_x = line[istart][0]
            range_end_x = line[ilast][1]
            assert range_end_x <= size[1]
            range_width = range_end_x - range_start_x
            module_width = range_width / 7
            likelihood = 1
            for i, (b, e) in enumerate(line[istart: iend]):
                width = e - b
                expected_width = module_width * modules_parts[i]
                assert width >= 0
                likelihood *= 1 - abs(width - expected_width) / max(width, expected_width)
            if likelihood > threshold:
                result[y, range_start_x:range_end_x] = 1
                all_modules_widths.append(module_width)
    success = len(all_modules_widths) > 0
    if not success:
        all_modules_widths = [1]
    return result, np.mean(all_modules_widths), success


def apply_mask_to_g(im, mask):
    im = np.array(im)
    r = im[:, :, 2]
    r[mask] = 0
    im[:, :, 2] = r

    g = im[:, :, 1]
    g[mask] = 255
    im[:, :, 1] = g

    return im


def len_sq_points(a, b):
    ax, ay = a
    bx, by = b
    return (bx - ax) * (bx - ax) + (by - ay) * (by - ay)


def check_pifagor(a, b, c):
    sum_katet = a + b
    not_likelyhood = abs(sum_katet - c) / max(sum_katet, c)
    if not_likelyhood < 0.15:
        return abs(a - b) / max(a, b) < 0.2
    return False


def check_pifagor_all(a, b, c):
    if check_pifagor(a, b, c):
        return True
    if check_pifagor(a, c, b):
        return True
    if check_pifagor(b, c, a):
        return True
    return False


def check_by_stats(stats, th_area_upper, th_area_lower, idx):
    if idx == 0:
        return False
    if stats[cv2.CC_STAT_AREA] > th_area_upper:
        return False
    if stats[cv2.CC_STAT_AREA] < th_area_lower:
        return False
    ar = stats[cv2.CC_STAT_HEIGHT] / stats[cv2.CC_STAT_WIDTH]
    if ar < 0.74 or ar > 1.28:
        return False
    rect_area = stats[cv2.CC_STAT_HEIGHT] * stats[cv2.CC_STAT_WIDTH]
    if stats[cv2.CC_STAT_AREA] < (rect_area - 5):
        return False
    return True


def process_image(im, dbg_name=None):
    binim = binarize(im)
    rle = to_rle(binim)
    rle_transp = to_rle(binim.T)
    h, w = binim.shape

    if dbg_name is not None:
        cv2.imwrite(f'dbg/{dbg_name}_bin.png', binim)

    was_match = False
    th_area_lower = 36
    for th in [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.09, 0.075, 0.0625, 0.5]:
        heatmap, module_width, success = heat_map_likelihood((h, w), rle, th)
        if not success:
            continue
        heatmap_transp = heat_map_likelihood((w, h), rle_transp, th)[0].T
        global_heatmap = heatmap * heatmap_transp
        orig_global_heatmap = global_heatmap
        module_width = max(int(module_width / 2), 1)
        kernel = np.ones((module_width, module_width), np.uint8)
        global_heatmap = cv2.morphologyEx(global_heatmap, cv2.MORPH_OPEN, kernel, iterations=1)
        kernel = np.ones((2 * module_width, 2 * module_width), np.uint8)
        global_heatmap = cv2.morphologyEx(global_heatmap, cv2.MORPH_CLOSE, kernel, iterations=1)

        if dbg_name is not None:
            cv2.imwrite(f'dbg/{dbg_name}_{th}_proc.png', apply_mask_to_g(im, global_heatmap > 0))
            cv2.imwrite(f'dbg/{dbg_name}_{th}_orig.png', apply_mask_to_g(im, orig_global_heatmap > 0))

        # num_labels, labels_im = cv2.connectedComponents(global_heatmap)
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(global_heatmap)
        # print(f'Level {th} found {num_labels}')
        if num_labels < 3:
            continue
        # print(centroids)
        res = []
        th_area_upper = h * w * 0.02
        areas = []
        index_mapping = np.argsort(stats[:, cv2.CC_STAT_AREA])
        index_mapping = index_mapping[-19:]
        evaluated_count = len(index_mapping)
        for j1 in range(0, evaluated_count):
            i1 = index_mapping[j1]
            if not check_by_stats(stats[i1], th_area_upper, th_area_lower, i1):
                continue
            for j2 in range(j1 + 1, evaluated_count):
                i2 = index_mapping[j2]
                if not check_by_stats(stats[i2], th_area_upper, th_area_lower, i2):
                    continue
                for j3 in range(j2 + 1, evaluated_count):
                    i3 = index_mapping[j3]
                    if not check_by_stats(stats[i3], th_area_upper, th_area_lower, i3):
                        continue
                    len12 = len_sq_points(centroids[i1], centroids[i2])
                    len23 = len_sq_points(centroids[i2], centroids[i3])
                    len13 = len_sq_points(centroids[i1], centroids[i3])
                    # print(f'{len12} {len23} {len13}')
                    if check_pifagor_all(len12, len13, len23):
                        res.append(i1)
                        res.append(i2)
                        res.append(i3)
                        areas.append(stats[i1, cv2.CC_STAT_AREA])
                        areas.append(stats[i2, cv2.CC_STAT_AREA])
                        areas.append(stats[i3, cv2.CC_STAT_AREA])
                        # print(f'Res {i1} {i2} {i3} on level {th}')
        if len(res) > 0:
            if was_match:
                for r in res:
                    im = apply_mask_to_g(im, labels_im == r)
                return im
            else:
                was_match = True
                th_area_lower = np.mean(areas) * 0.7
            # return labels_im[res[0]]

    print('Fail!')
    return im


def process_dir(path):
    files = os.listdir(path)
    output_dir = 'output2'
    try:
        os.mkdir(output_dir)
    except:
        print('Out dir exists')
    for file in files:
        print(f'Processing {file}')
        input_path = os.path.join(path, file)
        img = cv2.imread(input_path)
        result = process_image(img)
        # result = process_image(img, dbg_name=file)
        cv2.imwrite(os.path.join(output_dir, file), result)


def main():
    process_dir('TestSet2')


if __name__ == '__main__':
    main()
