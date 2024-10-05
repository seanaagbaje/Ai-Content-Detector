import tkinter as tk
from tkinterdnd2 import TkinterDnD, DND_FILES
from tkinter import filedialog, messagebox
import threading
from unified_detector import UnifiedDetector

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("AI-Generated Content, Deepfake, and Fake Account Detection")
        self.detector = UnifiedDetector()

        # Create layout frames for better organization
        self.main_frame = tk.Frame(self.root, padx=20, pady=20)
        self.main_frame.pack(fill="both", expand=True)

        # Text Detection Section
        self.text_frame = tk.LabelFrame(self.main_frame, text="AI-Generated Text Detection", padx=20, pady=10)
        self.text_frame.pack(pady=10, fill="both", expand=True)
        self.text_entry = tk.Entry(self.text_frame, width=50)
        self.text_entry.pack(side=tk.LEFT, padx=10, pady=5)
        self.text_button = tk.Button(self.text_frame, text="Analyze Text", command=self.start_analyze_text)
        self.text_button.pack(side=tk.LEFT, padx=10, pady=5)
        self.text_result_label = tk.Label(self.text_frame, text="", fg="blue")
        self.text_result_label.pack(side=tk.LEFT, padx=10, pady=5)

        # Deepfake Detection Section (Images & Videos with Drag and Drop)
        self.deepfake_frame = tk.LabelFrame(self.main_frame, text="Deepfake Detection (Image/Video)", padx=20, pady=10)
        self.deepfake_frame.pack(pady=10, fill="both", expand=True)

        self.drop_label = tk.Label(self.deepfake_frame, text="Drag and drop an image or video file here", fg="blue", relief="solid", padx=10, pady=10)
        self.drop_label.pack(pady=5, fill="both")
        self.drop_label.drop_target_register(DND_FILES)
        self.drop_label.dnd_bind('<<Drop>>', self.on_drop_file)

        self.file_path_label = tk.Label(self.deepfake_frame, text="No file selected", fg="blue")
        self.file_path_label.pack(pady=5)

        self.deepfake_button_img = tk.Button(self.deepfake_frame, text="Select Image", command=self.select_image)
        self.deepfake_button_img.pack(side=tk.LEFT, padx=10, pady=5)
        self.deepfake_button_vid = tk.Button(self.deepfake_frame, text="Select Video", command=self.select_video)
        self.deepfake_button_vid.pack(side=tk.LEFT, padx=10, pady=5)
        self.clear_button = tk.Button(self.deepfake_frame, text="Clear Selection", command=self.clear_selection)
        self.clear_button.pack(side=tk.LEFT, padx=10, pady=5)

        self.deepfake_result_label = tk.Label(self.deepfake_frame, text="", fg="blue")
        self.deepfake_result_label.pack(pady=5)

        # Fake Account Detection Section
        self.fake_account_frame = tk.LabelFrame(self.main_frame, text="Fake Account Detection", padx=20, pady=10)
        self.fake_account_frame.pack(pady=10, fill="both", expand=True)

        tk.Label(self.fake_account_frame, text="Username:").grid(row=0, column=0, sticky=tk.W)
        self.username_entry = tk.Entry(self.fake_account_frame, width=30)
        self.username_entry.grid(row=0, column=1, pady=5)

        tk.Label(self.fake_account_frame, text="Followers:").grid(row=1, column=0, sticky=tk.W)
        self.followers_entry = tk.Entry(self.fake_account_frame, width=30)
        self.followers_entry.grid(row=1, column=1, pady=5)

        tk.Label(self.fake_account_frame, text="Following:").grid(row=2, column=0, sticky=tk.W)
        self.following_entry = tk.Entry(self.fake_account_frame, width=30)
        self.following_entry.grid(row=2, column=1, pady=5)

        tk.Label(self.fake_account_frame, text="Number of Posts:").grid(row=3, column=0, sticky=tk.W)
        self.posts_entry = tk.Entry(self.fake_account_frame, width=30)
        self.posts_entry.grid(row=3, column=1, pady=5)

        tk.Label(self.fake_account_frame, text="Bio:").grid(row=4, column=0, sticky=tk.W)
        self.bio_entry = tk.Entry(self.fake_account_frame, width=30)
        self.bio_entry.grid(row=4, column=1, pady=5)

        tk.Label(self.fake_account_frame, text="Profile Picture (default/custom):").grid(row=5, column=0, sticky=tk.W)
        self.profile_picture_entry = tk.Entry(self.fake_account_frame, width=30)
        self.profile_picture_entry.grid(row=5, column=1, pady=5)

        self.fake_account_button = tk.Button(self.fake_account_frame, text="Check Account", command=self.start_fake_account_check)
        self.fake_account_button.grid(row=6, columnspan=2, pady=10)

        self.fake_account_clear_button = tk.Button(self.fake_account_frame, text="Clear", command=self.clear_fake_account_fields)
        self.fake_account_clear_button.grid(row=7, columnspan=2, pady=5)

        self.fake_account_result_label = tk.Label(self.fake_account_frame, text="", fg="blue")
        self.fake_account_result_label.grid(row=8, columnspan=2)

    def clear_selection(self):
        """Clears the file selection for deepfake detection."""
        self.file_path_label.config(text="No file selected")
        self.deepfake_result_label.config(text="")
        self.drop_label.config(text="Drag and drop an image or video file here", fg="blue")

    def clear_fake_account_fields(self):
        """Clears all input fields in the fake account detection section."""
        self.username_entry.delete(0, tk.END)
        self.followers_entry.delete(0, tk.END)
        self.following_entry.delete(0, tk.END)
        self.posts_entry.delete(0, tk.END)
        self.bio_entry.delete(0, tk.END)
        self.profile_picture_entry.delete(0, tk.END)
        self.fake_account_result_label.config(text="")

    def on_drop_file(self, event):
        """Handles drag-and-drop files."""
        file_path = event.data
        self.file_path_label.config(text=file_path)
        if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            self.start_analyze_image()
        elif file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mpeg', '.mpg')):
            self.start_analyze_video()
        else:
            messagebox.showwarning("File Error", "Unsupported file type. Please upload an image or video file.")

    # Text detection analysis
    def analyze_text(self):
        text_input = self.text_entry.get()
        if not text_input:
            messagebox.showwarning("Input Error", "Please enter some text to analyze.")
            return

        self.text_result_label.config(text="Analyzing text...", fg="black")
        predicted_label, confidence = self.detector.predict_text(text_input)
        verdict = "AI-generated" if predicted_label == 1 else "Human-written"
        self.text_result_label.config(text=f"{verdict} (Confidence: {confidence:.2f})", fg="green" if predicted_label == 0 else "red")

    def start_analyze_text(self):
        threading.Thread(target=self.analyze_text).start()

    # Image detection analysis
    def select_image(self):
        img_path = filedialog.askopenfilename(initialdir="/", title="Select Image",
                                              filetypes=(("jpeg files", "*.jpg"), ("png files", "*.png"), ("all files", "*.*")))
        if img_path:
            self.file_path_label.config(text=img_path)
            self.start_analyze_image()

    def analyze_image(self):
        img_path = self.file_path_label.cget("text")
        self.deepfake_result_label.config(text="Analyzing image...", fg="black")
        prediction, confidence = self.detector.predict_image(img_path)
        verdict = "Deepfake" if prediction == 1 else "Real"
        self.deepfake_result_label.config(text=f"{verdict} (Confidence: {confidence:.2f})", fg="red" if prediction == 1 else "green")

    def start_analyze_image(self):
        threading.Thread(target=self.analyze_image).start()

    # Video detection analysis
    def select_video(self):
        vid_path = filedialog.askopenfilename(initialdir="/", title="Select Video",
                                              filetypes=(("mp4 files", "*.mp4"), ("avi files", "*.avi"), ("mov files", "*.mov"),
                                                         ("mpeg files", "*.mpeg"), ("mpg files", "*.mpg"), ("all files", "*.*")))
        if vid_path:
            self.file_path_label.config(text=vid_path)
            self.start_analyze_video()

    def analyze_video(self):
        vid_path = self.file_path_label.cget("text")
        self.deepfake_result_label.config(text="Processing video...", fg="black")
        avg_prediction, avg_confidence = self.detector.predict_video(vid_path)
        verdict = "Deepfake" if avg_prediction >= 0.5 else "Real"
        self.deepfake_result_label.config(text=f"{verdict} (Avg Confidence: {avg_confidence:.2f})", fg="red" if avg_prediction >= 0.5 else "green")

    def start_analyze_video(self):
        threading.Thread(target=self.analyze_video).start()

    # Fake account detection analysis
    def analyze_fake_account(self):
        account_data = {
            'username': self.username_entry.get(),
            'profile_picture': self.profile_picture_entry.get(),
            'followers': int(self.followers_entry.get()),
            'following': int(self.following_entry.get()),
            'posts': int(self.posts_entry.get()),
            'bio': self.bio_entry.get(),
            'url_in_bio': ''
        }
        self.fake_account_result_label.config(text="Checking account...", fg="black")
        verdict = self.detector.detect_fake_account(account_data)
        self.fake_account_result_label.config(text=verdict, fg="red" if verdict == "Fake Account" else "green")

    def start_fake_account_check(self):
        threading.Thread(target=self.analyze_fake_account).start()

if __name__ == "__main__":
    root = TkinterDnD.Tk()  # Enable TkinterDnD for drag-and-drop
    app = App(root)
    root.mainloop()
