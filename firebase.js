const firebaseConfig = {
    apiKey: "AIzaSyCEnqohf95KsGaXSUFd2TUTkEv7tM3mpCo",
    authDomain: "enhancer-b0580.firebaseapp.com",
    databaseURL: "https://enhancer-b0580-default-rtdb.firebaseio.com",
    projectId: "enhancer-b0580",
    storageBucket: "enhancer-b0580.appspot.com",
    messagingSenderId: "903561426277",
    appId: "1:903561426277:web:624db060b0f2be0220dd8c"
};


firebase.initializeApp(firebaseConfig);
  firebase.analytics();

// let's code 
var datab  = firebase.database().ref('data');
function UserRegister(){
var email = document.getElementById('email').value;
var password = document.getElementById('password').value;
firebase.auth().createUserWithEmailAndPassword(email,password).then(function(){
    
}).catch(function (error){
    var errorcode = error.code;
    var errormsg = error.message;
});
}
const auth = firebase.auth();
function SignIn(){
    var email = document.getElementById('email').value;
    var password = document.getElementById('password').value;
    const promise = auth.signInWithEmailAndPassword(email,password);
    promise.catch( e => alert(e.msg));
    window.open("https://www.google.com","_self");
}
document.getElementById('form').addEventListener('submit', (e) => {
    e.preventDefault();
    var userInfo = datab.push();
    userInfo.set({
        name: getId('fname'),
        email : getId('eemail'),
        password : getId('lpassword')
    });
    alert("Successfully Signed Up");
    console.log("sent");
    document.getElementById('form').reset();
});
function  getId(id){
    return document.getElementById(id).value;
}