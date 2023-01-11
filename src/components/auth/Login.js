import React, { useState } from "react";
import "./Login.css";
import validator from 'validator';
import ArrowForwardIosIcon from "@material-ui/icons/ArrowForwardIos";
import { auth, provider } from "../../firebase";
import { Avatar, Button, Input } from "@material-ui/core";
import db from "../../firebase";
import Modal from "react-modal";
Modal.setAppElement("#root");
function Login() {
  const [IsmodalOpen, setIsModalOpen] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [remail, setREmail] = useState('');
  const [rpassword, setRPassword] = useState('');
  const [name, setName] = useState('');
 
  

  var a="Save";
  const signIn = () => {
    auth.signInWithPopup(provider).catch((e) => {
      alert(e.message);
    });
  };

  const handleSignIn = (e) => {
    e.preventDefault();

    auth
      .signInWithEmailAndPassword(email, password)
      .then((auth) => {
        console.log(auth);
      })
      .catch((e) => alert(e.message));
  };
  const handleCancel=()=>{
    setIsModalOpen(false);
    
  }

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log(name.length>8);
    var trim_name= name.replace(" ","");
    console.log(trim_name);
    if(name==='' || remail === '' || rpassword === '') {
      alert("Please Enter all the Fields");
    } 
    else if(!(name.length>8 && name.length<25)){
      alert("Enter a valid name of length bewteen 8 and 25");
    }
    else if (!validator.isEmail(remail)) {
      alert('Enter valid Email!');
    } 
    else if(!validator.isAlpha(trim_name)){
           alert("Enter a valid name");
    }
    else if(!validator.isStrongPassword(rpassword)){
      alert("Enter a valid password");
    }
    else {
     

    auth
      .createUserWithEmailAndPassword(remail, rpassword)
      .then((auth) => {
        if (auth) {
          console.log(auth);
          db.collection("User").add({
            email:remail,
            password:rpassword,
            notification: a,
            name:name,
          });
        }
      })
      .catch((e) => alert(e.message));
      
    }
  };

  return (
    <div className="login">
      <div className="login__container">
        <div className="login__logo">
          <div className="project-name">
        CURIOSITY
        </div>
          </div>
        <div className="login__desc">
          <h3></h3>
        </div>
        
        <div className="login__auth">
          
          <div className="login__authOptions">
          <div className="login__authDesc">
              <p>
                <span style={{ color: "orangered", cursor: "pointer" }}>
                  
                </span>
                <span style={{ color: "orangered", cursor: "pointer" }}>
                  &nbsp; 
                </span>
                
                <span style={{ color: "orangered", cursor: "pointer" }}>
                  
                </span>
                
              </p>
            </div>
            <div className="login__authOption">
              <img
                className="login__googleAuth"
                src="https://media-public.canva.com/MADnBiAubGA/3/screen.svg"
                alt=""
              />
             
              
              <p onClick={signIn}>Continue With Google</p>
            </div>
            
            
          </div>
          <div className="login__emailPass">
          
            <div className="login__label">
            <h2 class="neonText"> <div class ="Login">
            
            
           <h1>LOGIN</h1>
            
            
            </div></h2> 

            </div>
            <div className="login__inputFields">
              <div className="login__inputField">
                <input
                  value={email} class="P"
                  onChange={(e) => setEmail(e.target.value)}
                  type="text"
                  placeholder="Email"
                />
              </div>
              <div className="login__inputField">
                <input
                  value={password} class ="P"
                  onChange={(e) => setPassword(e.target.value)}
                  type="password"
                  placeholder="Password"
                />
              </div>
            </div>
            <div className="login__forgButt">
              <small>Forgot Password?</small>
              <button onClick={handleSignIn}>Login</button>
            </div>

            
            

            <button onClick={() => setIsModalOpen(true)}>Register</button>
            <Modal
          isOpen={IsmodalOpen}
          onRequestClose={() => setIsModalOpen(false)}
          shouldCloseOnOverlayClick={false}
          style={{
            overlay: {
              width: 700,
              height: 600,
              backgroundColor: "rgba(0,0,0,0.8)",
              zIndex: "1000",
              top: "50%",
              left: "50%",
              marginTop: "-300px",
              marginLeft: "-350px",
            },
          }}
        >
         
          <div className="modal__title">
            <h5>Register</h5>
          </div>

          <form>
       
        <label className="label">Name</label>
        <span className="tab"></span>
        <input  onChange={(e) => setName(e.target.value)} className="input" 
        value={name} type="text" />
        <br></br>
        <br></br>
        <label  className="label">Email</label>
        <span className="tab2"></span>
        <input onChange={(e) => setREmail(e.target.value)} className="input" 
          value={remail} type="email" />
          <br></br>
          <br></br>
        <label className="label">Password</label>
        <span className="tab3"></span>
        <input onChange={(e) => setRPassword(e.target.value)} className="input" 
          value={rpassword} type="password" />
        <br></br>
        
      
        
          <div className="modal__buttons_Login">
            <button className="cancle_Login" onClick={handleCancel}>
              Cancel
            </button>
            <button type="submit" onClick={handleSubmit} className="register_Login" >
              Register
            </button>
          </div>
          </form>
        </Modal>
          </div>
        </div>
        <div className="login__lang">
          <p>WELCOME</p>

        </div>
        <div className="login__footer">
          
        </div>
      </div>
    </div>
  );
}

export default Login;