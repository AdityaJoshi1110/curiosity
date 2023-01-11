import React, { useEffect, useState } from "react";
import PeopleAltOutlinedIcon from "@material-ui/icons/PeopleAltOutlined";
import SearchIcon from "@material-ui/icons/Search";
import LanguageIcon from "@material-ui/icons/Language";
import { Home, LocationOn, Notifications } from '@material-ui/icons';
import Modal from "react-modal";
import Select from "react-select";
import "./QHeader.css";
import { Avatar, Button, Input } from "@material-ui/core";
import { useSelector } from "react-redux";
import { selectUser } from "../features/userSlice";
import db, { auth } from "../firebase";
import { ExpandMore, Link } from "@material-ui/icons";
import firebase from "firebase";
import {Dropdown} from 'react-bootstrap';

Modal.setAppElement("#root");

function QHeader() {
  const user = useSelector(selectUser);

  const [IsmodalOpen, setIsModalOpen] = useState(false);
  const [input, setInput] = useState("");
  const [inputUrl, setInputUrl] = useState("");
  const questionName = input;
  const [not,setNotifications]=useState("Save");
   var b;
   var a;
  useEffect(() => {
    db.collection("User").where("email","==",user.email)
      .onSnapshot((snapshot) =>                                     
        setNotifications(                                                      
          snapshot.docs.map((doc) => ({
            Uid: doc.id,
            Uemails: doc.data(),
          }))
        )
      );
  }, []);
   

  var categoryList=[
    {
      value:1,
      label:"General"
    },
    {
      value:2,
      label:"Business"
    },
    {
      value:3,
      label:"Computer Science"
    },
    {
      value:4,
      label:"Music"
    },
    {
      value:5,
      label:"Science"
    },
    {
      value:6,
      label:"Health"
    },
    {
      value:7,
      label:"Movies"
    },
    {
      value:8,
      label:"Technology"
    },
    {
      value:9,
      label:"Education"
    }
  ];
  const[category,catValue]=useState(categoryList.label);
  const catHandler=e=>{
    if(e!="" || e!=null)
    {
      catValue(e.label);
    }
     
  }
  
  const getNotification=()=>{
    console.log("Hello");
    not.map(({Uid,Uemails})=>{
     
    });
  }

  const handleCancel=()=>{
    setIsModalOpen(false);
    setInput("");
    setInputUrl("");
  }
  const handleQuestion = (e) => {
    if(input=="" || category=="" || category==null)
    {
       alert("Invalid Field");
       
    }
    else
    {
      setIsModalOpen(false);
      e.preventDefault();
      // setIsModalOpen(false);
   
       if (questionName) {
         db.collection("questions").add({
           user: user,
           question: input,
           imageUrl: inputUrl,
           category:category,
           timestamp: firebase.firestore.FieldValue.serverTimestamp(),
           likes:0,
           dislikes:0,
         });
    }
   
      
    }

    setInput("");
    setInputUrl("");
  };

  return (
    <div className="QHeader">
      <div className="QHeader__logo2">
       
        <img src="https://mms.businesswire.com/media/20210823005374/en/901163/22/Curiosity_Wordmark_Color_Pos.jpg"></img>
     
      </div>
      <div className="QHeader__logo1">
       
        <img src="https://motivatedandfree.files.wordpress.com/2020/05/ssizq5my0eba5y3r2m.gif"></img>
     
      </div>
      
      
        

      <div className="QHeader__Rem">
        <div className="QHeader__avatar">
          <Avatar
            onClick={() => auth.signOut()}
            className="Avatar"
            src={
              user.photo
                ? user.photo
                : ""
            }
          />
        </div>
        
        {/*
         <div classname="notification">
         <Dropdown  onClick={getNotification} >
        <Dropdown.Toggle variant="success" id="dropdown-basic">
        <Notifications />
        
   
 </Dropdown.Toggle>
  <Dropdown.Menu >
    <Dropdown.Item href="#/action-1"> </Dropdown.Item>
    <Dropdown.Item href="#/action-2">{a}</Dropdown.Item>
    <Dropdown.Item href="#/action-3">Something else</Dropdown.Item>
  </Dropdown.Menu>
        </Dropdown>
       </div>
        */}
        <Button onClick={() => setIsModalOpen(true)}>Add Question</Button>
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
            <h5>Add Question</h5>
          </div>
          <div className="modal__info">
            <Avatar
              className="avatar"
              src={
                user.photo
                  ? user.photo
                  : ""
              }
            />
            <p>{user.disPlayName ? user.disPlayName : user.email} asked</p>
            <div className="modal__scope">
              
              
              
            </div>
          </div>
          <div className="modal__Field">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              type="text"
              placeholder="Start your question with 'What', 'How', 'Why', etc. "
            />
            <div className="modal__fieldLink">
              <Link />
              <input
                value={inputUrl}
                onChange={(e) => setInputUrl(e.target.value)}
                type="text"
                placeholder="Optional: inclue a link that gives context"
              ></input>
            </div>
            <div className="modal_category1">
            <div class="category-content">
            <Select options={categoryList}  onChange={catHandler} required>
            </Select>
            </div>
            </div>
          </div>
          <div className="modal__buttons">
            <button className="cancle" onClick={handleCancel}>
              Cancel
            </button>
            <button type="sumbit" onClick={handleQuestion} className="add" >
              Add Question
            </button>
          </div>
        </Modal>
      </div>
    </div>
  );
}

export default QHeader;