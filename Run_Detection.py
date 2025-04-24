import sys
import os
import glob
import tkinter as tk
from tkinter import messagebox
import tkinter.ttk as ttk
import yagmail  # Ensure yagmail is installed

class Detection:
    def __init__(self, top=None):
        self.index = 0
        self.ChallanImg = -1
        self.listFiles = glob.glob('output/*.txt')
        
        if not self.listFiles:
            messagebox.showerror("Error", "No files found in output directory.")
            return
        
        # Initialize email sender (Replace with a secure method)
        self.sender_email = "biradarjanak33@gmail.com"
        self.yag = yagmail.SMTP(self.sender_email, os.getenv("EMAIL_PASSWORD"))

        def sendChallan(event):
            try:
                helmet = self.cmbHelmetStatus.get()
                signal = self.cmbSignal.get()
                plateNumber = self.txtPlateNumber.get()
                noOfPassengers = self.txtPassengers.get()
                amount = 0
                challanBreaked = ''
                
                if signal.lower() == "breaked":
                    amount += 500
                    challanBreaked += '<br><br><b>Signal Breaked (500 Rs)</b>'
                
                if helmet == "Not Wearing":
                    amount += 500
                    challanBreaked += '<br><br><b>Not Wearing Helmet (500 Rs)</b>'
                
                if noOfPassengers.isdigit() and int(noOfPassengers) >= 3:
                    amount += 1000
                    challanBreaked += '<br><br><b>3 Passengers in Vehicle (1000 Rs)</b>'
                
                message = f"Hi, User<br><br>You violated the following rules: {challanBreaked}<br><br>Challan Amount: <b>{amount} Rs.</b>"
                data = open(self.listFiles[self.ChallanImg]).read().split('\n')
                self.yag.send(to='aniketshekokar92@gmail.com', subject="Traffic Rule Challan", contents=message, attachments=data[0])
                messagebox.showinfo('Challan Sent', 'Challan email sent successfully.')
            except Exception as e:
                messagebox.showerror('Error', f'Failed to send mail: {str(e)}')

        def showOutput(event):
            try:
                if self.index >= len(self.listFiles):
                    self.index = 0
                    self.ChallanImg = -1
                    messagebox.showinfo('End', 'No more images to display.')
                    return
                
                data = open(self.listFiles[self.index]).read().split('\n')
                self.img = tk.PhotoImage(file=data[0])
                self.lblRider.configure(image=self.img)
                
                self.cmbHelmetStatus.set("Not Found" if data[1] == "None" else ("Wearing" if data[1] == "True" else "Not Wearing"))
                self.cmbPlateStatus.set("Not Found" if data[2] == "None" else ("Visible" if data[2] == "True" else "Not Visible"))
                self.txtPlateNumber.delete(0, "end")
                self.txtPlateNumber.insert(0, data[3] if data[3] != "None" else "")
                self.txtPassengers.delete(0, "end")
                self.txtPassengers.insert(0, data[4])
                self.cmbSignal.set("Breaked" if "Breaked" in data[5] else "Not Breaked")
                
                self.index += 1
                self.ChallanImg += 1
            except Exception as e:
                messagebox.showinfo('Error', f'Failed to load image: {str(e)}')

        def close(event):
            top.destroy()
        
        # GUI Configuration
        top.geometry("800x600")
        top.title("Helmet Detection System")
        
        self.lblRider = tk.Label(top, text="Image will be displayed here", bg="#d2d2d2")
        self.lblRider.pack()
        
        frame = tk.Frame(top)
        frame.pack()
        
        self.cmbHelmetStatus = ttk.Combobox(frame, values=['Wearing', 'Not Wearing', 'Not Found'])
        self.cmbHelmetStatus.pack()
        
        self.cmbPlateStatus = ttk.Combobox(frame, values=['Visible', 'Not Visible', 'Not Found'])
        self.cmbPlateStatus.pack()
        
        self.txtPlateNumber = tk.Entry(frame)
        self.txtPlateNumber.pack()
        
        self.txtPassengers = tk.Entry(frame)
        self.txtPassengers.pack()
        
        self.cmbSignal = ttk.Combobox(frame, values=['Breaked', 'Not Breaked'])
        self.cmbSignal.pack()
        
        self.btnSendChallan = tk.Button(frame, text="Send Challan")
        self.btnSendChallan.pack()
        self.btnSendChallan.bind('<Button-1>', sendChallan)
        
        self.btnNextImage = tk.Button(frame, text="Next Image")
        self.btnNextImage.pack()
        self.btnNextImage.bind('<Button-1>', showOutput)
        
        self.btnExit = tk.Button(frame, text="EXIT", command=top.destroy, bg="#d8368c", fg="white")
        self.btnExit.pack()
        
        showOutput(None)

if __name__ == "__main__":
    root = tk.Tk()
    app = Detection(root)
    root.mainloop()
