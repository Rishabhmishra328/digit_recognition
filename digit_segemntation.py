import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch as ss
import cv2
import os

def main():

    # loading digit image
    img = cv2.imread('digit_color.jpg')

    if not os.path.exists('./segments'):
        os.makedirs('./segments')


    # perform selective search
    img_lbl, regions = ss.selective_search(img, scale=200, sigma=0.9, min_size=50)

    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 1000:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        if w / (h + 0.000001) > 1.2 or h / (w + 0.000001) > 1.2:
            continue
        candidates.add(r['rect'])

    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    image_name = 1
    for x, y, w, h in candidates:
        rect = mpatches.Rectangle((x, y), w, h, fill=False, edgecolor='green', linewidth=1)
        crop_dist = w if w<h else h
        cropped = img[x:x+crop_dist, y:(y+crop_dist)]

        if(cropped.shape[0] == cropped.shape[1]):
            cropped = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
            cropped = cv2.resize(cropped, (28,28), interpolation = cv2.INTER_AREA)
            cv2.imwrite('./segments/' + str(image_name) + '.jpg', cropped)
            image_name += 1
        ax.add_patch(rect)

    plt.show()

if __name__ == "__main__":
    main()