from flask import Flask, url_for, request, render_template
app = Flask(__name__)


@app.route("/")
def hello():
    return "hello world!"


@app.route("/index")
def index():
    #return "this is index"
    return render_template('index.html')


@app.route("/user/<username>")
def show_user_profile(username):
    #return "user : %s" % username
    mydic = {'key1':111, 'key2':222}
    return render_template('user.html', name=username, dic=mydic)


@app.route("/post/<int:post_id>")
def show_post(post_id):
    return "post : %s" % post_id


@app.route("/url/")
def get_url():
    return url_for("show_post", post_id=2)


@app.route("/url2/")
def get_url2():
    return url_for("static", filename="css/style.css")


@app.route("/login/", methods=['POST', 'GET', 'PUT'])
def login():
    if request.method == "GET":
        return "get"
    elif request.method == "PUT":
        return "put"
    elif request.method == "POST":
        return "post"


if __name__ == "__main__":
    app.debug = True
    app.run(debug=True)