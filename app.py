from flask import Flask, render_template, url_for, request, redirect, flash
from register import register_yourself
from mark_attendance import mark_your_attendance


app = Flask(__name__)
app.secret_key = 'my secret key'      #Nothing important, type anything, just for flashing


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/', methods=['POST'])
def home_after_registration():
    id = request.form['Student_id']
    register_yourself(id)
    flash("Registration Successful")
    return render_template("index.html")


@app.route('/registration', methods=['GET', 'POST'])
def registration():
    return render_template("register.html")


@app.route('/attendance', methods=['GET', 'POST'])
def attendance():
    marked = mark_your_attendance()
    if(marked == True):
        flash("Attendence Marked Successfully")
    else:
        flash("You are not registered yet")

    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug = True)
