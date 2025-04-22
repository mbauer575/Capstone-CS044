import cv2, numpy as np, os, sys

def getImages(image_list, directory):
    extensions = ('.jpg', '.jpeg', '.png')           # add the missing dot
    for file in os.listdir(directory):
        if file.lower().endswith(extensions):
            image_list.append(file)

def main():
    #Get command line arguments
    if len(sys.argv)!=6:
        print('Incorrect amount of arguments')
        return
    
    input_dir=sys.argv[1]
    augmented_dir=sys.argv[2]
    brightness_up=float(sys.argv[3])
    brightness_down=float(sys.argv[4])
    blur_kernal=int(sys.argv[5])

    if blur_kernal%2!=1:
        print("Blur parameter must be an odd integer")
        return

    if not os.path.exists('../'+input_dir):
        print('Input directory not found')
        print('current dir'+os.getcwd())
        return

    # create the user-specified output folder
    if not os.path.exists(augmented_dir):
        os.makedirs(augmented_dir)

    image_list = []
    getImages(image_list, input_dir)

    for image in image_list:
        # load from the real input_dir
        img = cv2.imread(os.path.join(input_dir, image), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Failed to load {image}")
            continue

        # blur
        blur = cv2.GaussianBlur(img, (blur_kernal, blur_kernal), 0)
        cv2.imwrite(os.path.join(augmented_dir, f'blur_{image}'), blur)

        #Create different brithness versions of images
        hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

        if brightness_down!=0:
            h_l,s_l,v_l=cv2.split(hsv)

            v_l=v_l.astype(np.float32)
            v_l*=1-brightness_down
            v_l=np.clip(v_l, 0, 255)
            v_l=v_l.astype(np.uint8)

            final_l=cv2.merge((h_l,s_l,v_l))
            export_l=cv2.cvtColor(final_l,cv2.COLOR_HSV2BGR)

            cv2.imwrite(os.path.join(augmented_dir, f'low_{image}'), export_l)

            #Create blurred versions of the brightened/dimmed images
            blur_l=cv2.GaussianBlur(export_l,(blur_kernal,blur_kernal),0)
            cv2.imwrite(os.path.join(augmented_dir, f'blur_low_{image}'), blur_l)

        if brightness_up!=0:
            h_h,s_h,v_h=cv2.split(hsv)

            v_h=v_h.astype(np.float32)
            v_h*=1+brightness_up
            v_h=np.clip(v_h, 0, 255)
            v_h=v_h.astype(np.uint8)

            final_h=cv2.merge((h_h,s_h,v_h))
            export_h=cv2.cvtColor(final_h,cv2.COLOR_HSV2BGR)

            cv2.imwrite(os.path.join(augmented_dir, f'high_{image}'), export_h)

            #Create blurred versions of the brightened/dimmed images
            blur_h=cv2.GaussianBlur(export_h,(blur_kernal,blur_kernal),0)
            cv2.imwrite(os.path.join(augmented_dir, f'blur_high_{image}'), blur_h)

if __name__ == "__main__":
    main()