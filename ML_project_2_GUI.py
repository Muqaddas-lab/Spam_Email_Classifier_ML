import tkinter as tk
from tkinter import messagebox
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample data to train a dummy model (replace this with your trained model later)
emails = [
    "Win a free iPhone now",  # spam
    "Earn money fast",         # spam
    "Let's meet for lunch",    # ham
    "Project deadline is tomorrow",  # ham
]
labels = [1, 1, 0, 0]  # 1 = spam, 0 = ham

# rain dummy model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(emails)
model = MultinomialNB()
model.fit(X, labels)

# GUI Setup
root = tk.Tk()
root.title("Spam Email Classifier")
root.geometry("500x400")
root.configure(bg="#8507da")

# Title Label
title = tk.Label(root, text="Spam Email Detector", font=("Arial", 20, "bold"), fg="#004080", bg="#f0f8ff")
title.pack(pady=20)

# Email Input
entry = tk.Text(root, height=10, width=50, font=("Arial", 12))
entry.pack(pady=10)

#  Result Label
result_label = tk.Label(root, text="", font=("Arial", 16, "bold"), bg="#f0f8ff")
result_label.pack(pady=10)

#  Button Function
def classify_email():
    email_text = entry.get("1.0", tk.END).strip()
    if not email_text:
        messagebox.showwarning("Input Error", "Please enter an email text.")
        return
    vec = vectorizer.transform([email_text])
    prediction = model.predict(vec)[0]
    result = "🚫 Spam" if prediction == 1 else "✅ Ham"
    result_color = "red" if prediction == 1 else "green"
    result_label.config(text=result, fg=result_color)

# Check Button
check_btn = tk.Button(root, text="Check Email", command=classify_email, font=("Arial", 14), bg="#004080", fg="white")
check_btn.pack(pady=10)

root.mainloop()
