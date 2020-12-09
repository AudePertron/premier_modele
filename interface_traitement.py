import tkinter as tk
from tkinter import PhotoImage
from tkinter import filedialog
colors = {
    'orange':'#f9690c',
    'noir':'#030100',
    'blanc' :'#f9f5f5'
}

class Interface:
    def __init__(self, master):
        self.master = master
        self.master.title("traitement données")
        self.master.geometry('1200x600')
        self.master.minsize(900, 600)
        self.master.configure(bg=colors['noir'])

        #logo
        logo = PhotoImage(file="C:/Users/utilisateur/Documents/microsoft_ia/traitement/logo.png")
        logo_f = tk.Label( image = logo)    ###frame nécéssaire?
        logo_f.pack(pady=10, padx=10, side='top') ###vérifier superposition avec titre


        # titre
        titre = tk.Label(self.master, text='Bienvenue sur la magnifique interface de traitement des données de Aude', 
                        font=("Helvetica", 20), bg=colors['orange'] , fg=colors['noir'])
        titre.pack(fill='x', side='top')

        #frame boutons
        self.frame_boutons = tk.Frame(self.master, bg=colors['blanc'])
        self.frame_boutons.pack(pady=25)

        #frame menu
        self.frame_menu = tk.Frame(self.master, bg=colors['noir'])
        self.frame_menu.pack()

        #fonction d'ajout de bouton
        fonctions = {'Affichage du fichier crédit': self.afficher_credit}

        for i, (key, value) in enumerate(fonctions.items()):
            ligne = tk.Button(self.frame_boutons, height=2, width=12, bg=colors['orange'], bd=0, font=(
                'Helvetica', '12'), text=key, command=self.browseFiles)
            ligne.grid(row=0, column=i, padx=5, ipadx=12)

    def browseFiles(self): 
            global filename
            filename = filedialog.askopenfilename(initialdir = "/", 
                                                title = "Select a File", 
                                                filetypes = (("CSV files", 
                                                                "*.csv*"), 
                                                            ("all files", 
                                                                "*.*"))) 
            self.openedFileLabel.configure(text=filename.split("/")[-1])





        #bouton afficher fichier crédit
    def afficher_credit(self):
        for widget in self.frame_menu.winfo_children():
            widget.pack_forget()

        credit_frame = tk.Frame(self.frame_menu, bg=color['blanc'])
        credit_frame.pack()

        credit= fonctions   ###importer fonction affichage?

        credit_label = tk.Label(credit_frame, text="credit", bg=colors['blanc'], font=(
            'Helvetica', '20', 'underline'))
        credit_label.grid(row=0, column=1, padx=50)

        for i, value in enumerate(credit, 1):
            label = tk.Label(credit_frame, text=value,
                            bg=colors['blanc'], font=('Helvetica', '12'))
            label.grid(row=i, column=1)
     

