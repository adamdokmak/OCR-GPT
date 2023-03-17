import cv2
import pyautogui
import pytesseract
from PIL import Image
import numpy as np
import openai
from pynput import keyboard
import matplotlib.pyplot as plt

plt.imshow(img)
plt.show()


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# def select_roi(img):
#     window_name = "Select ROI"
#     cv2.startWindowThread()
#     cv2.imshow(window_name, img)
#     cv2.waitKey(1)
#     region = select_roi(screenshot_np.copy())
#     roi = cv2.selectROI(window_name, img, fromCenter=False)
#     cv2.destroyWindow(window_name)
#
#     return roi


pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract' # Change this path to your tesseract executable path

openai.api_key = "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"

def get_snippet():
    screen_width, screen_height = pyautogui.size()
    screenshot = pyautogui.screenshot()
    screenshot_np = np.array(screenshot)
    screenshot_np = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
    region = select_roi(screenshot_np)
    cropped = screenshot_np[int(region[1]):int(region[1] + region[3]), int(region[0]):int(region[0] + region[2])]
    img = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    text = pytesseract.image_to_string(img)
    region = select_roi(screenshot_np)
    if region is None:
        print("Failed to get the ROI. Try again.")
        return None

    cv2.destroyAllWindows()

    return text

def send_to_chatgpt(text):
    try:
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=text,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.5
        )

        generated_text = response.choices[0].text.strip()
        return generated_text
    except Exception as e:
        print(f"Error: {e}")
        return None

def on_press(key):
    try:
        if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
            print("Select the snippet region")
            snippet_text = get_snippet()

            print(f"Captured text:\n{snippet_text}")

            print("Sending text to ChatGPT")
            chatgpt_response = send_to_chatgpt(snippet_text)

            if chatgpt_response:
                print(f"ChatGPT response:\n{chatgpt_response}")
            else:
                print("Error occurred while sending text to ChatGPT")
    except AttributeError:
        pass

print("Press Ctrl to start capturing a text snippet")

# Start the keyboard listener
with keyboard.Listener(on_press=on_press) as listener:
    listener.join()
