import { Avatar } from "@material-ui/core";
import React, { useEffect, useState } from "react";
import "./MyQuestionPost.css";
import ArrowUpwardOutlinedIcon from "@material-ui/icons/ArrowUpwardOutlined";
import ArrowDownwardOutlinedIcon from "@material-ui/icons/ArrowDownwardOutlined";
import { useDispatch, useSelector } from "react-redux";
import { selectUser } from "../features/userSlice";
import Modal from "react-modal";
import db from "../firebase";
import { selectQuestionId, setQuestionInfo } from "../features/questionSlice";
import firebase from "firebase";

function MyQuestionPost({ Id, question, imageUrl, timestamp, users}) {
  const user = useSelector(selectUser);
  const dispatch = useDispatch();
   var a=0;
  const [IsmodalOpen, setIsModalOpen] = useState(false);
  const questionId = useSelector(selectQuestionId);
  const [answer, setAnswer] = useState("");
  const [getAnswers, setGetAnswers] = useState([]);
  const [likes] = useState("");
  const [dislikes] = useState("");
  useEffect(() => {
   
    if (questionId) {
      db.collection("questions")
        .doc(questionId)
        .collection("answer")
        .orderBy("timestamp", "desc")
        .onSnapshot((snapshot) =>
          setGetAnswers(
            snapshot.docs.map((doc) => ({ id: doc.id, answers: doc.data() }))
          )
        );
    }
  }, [questionId]);

  const handledelete=()=>{
    
    dispatch(
      setQuestionInfo({
        questionId: Id,
        questionName: question,
      })
    )
    if(questionId)
    {
     db.collection("questions").doc(questionId).collection("answer").get().then(querySnapshot => {
      querySnapshot.docs.forEach(snapshot => {
          snapshot.ref.delete();
      })
      });
    }
      db.collection("questions").doc(questionId).delete();
    alert('Question Deleted');
    }
  


  const handleLikes1=()=>{
    dispatch(
    setQuestionInfo({
      questionId: Id,
      questionName: question,
    })
  )
  
     a=db.collection("questions").doc(questionId);  
     a.update({
       likes: firebase.firestore.FieldValue.increment(1)
     });
 }



  const handleAnswer1 = (e) => {
    e.preventDefault();

    if (questionId) {
      db.collection("questions").doc(questionId).collection("answer").add({
        user: user,
        answer: answer,
        questionId: questionId,
        timestamp: firebase.firestore.FieldValue.serverTimestamp(),
      });
    }
    console.log(questionId);
    setAnswer("");
    setIsModalOpen(false);
  };
  return (
    <div
      className="post"
      onClick={() =>
        dispatch(
          setQuestionInfo({
            questionId: Id,
            questionName: question,
          })
        )
      }
    >
      <div className="post__info">
        <Avatar
          src={
            users.photo
              ? users.photo
              : "https://images-platform.99static.com//_QXV_u2KU7-ihGjWZVHQb5d-yVM=/238x1326:821x1909/fit-in/500x500/99designs-contests-attachments/119/119362/attachment_119362573"
          }
        />
        <h4>{users.displayName ? users.displayName : users.email}</h4>
        <small>{new Date(timestamp?.toDate()).toLocaleString()}</small>
      </div>
      <div className="post__body">
        <div className="post_question1">
          <p>{question}</p>
          
         <button className="post_btnDelete" onDoubleClick={handledelete}>
          Delete
         </button>
         
          <button
            onClick={() => setIsModalOpen(true)
            }
            className="post_btnAnswer1"
          >
            Answer
          </button>
          
        
      
          
         
          <Modal
            isOpen={IsmodalOpen}
            onRequestClose={() => setIsModalOpen(false)}
            shouldCloseOnOverlayClick={false}
            style={{
              overlay: {
                width: 680,
                height: 550,
                backgroundColor: "rgba(0,0,0,0.8)",
                zIndex: "1000",
                top: "50%",
                left: "50%",
                marginTop: "-250px",
                marginLeft: "-350px",
              },
            }}
          >
            <div className="modal__question">
              <h1>{question}</h1>
              <p>
                asked by{" "}
                <span className="name">
                  {users.displayName ? users.displayName : users.email}
                </span>{" "}
                {""}
                on{" "}
                <span className="name">
                  {new Date(timestamp?.toDate()).toLocaleString()}
                </span>
              </p>
            </div>
            <div className="modal__answer">
              <textarea
                value={answer}
                onChange={(e) => setAnswer(e.target.value)}
                placeholder="Enter Your Answer"
                type="text"
              />
            </div>
            <div className="modal__button">
              <button className="cancle" onClick={() => setIsModalOpen(false)}>
                Cancel
              </button>
              <button type="sumbit" onClick={handleAnswer1} className="add">
                Add Answer
              </button>
            </div>
          </Modal>
        </div>
        <img src={imageUrl} alt="" />
        <div className="post_answer">
          {getAnswers.map(({ id, answers }) => (
            <p key={id} style={{ position: "relative", paddingBottom: "5px" }}>
              {Id === answers.questionId ? (
                <span>
                  {answers.answer}
                  <br />
                  <span
                    style={{
                      position: "absolute",
                      color: "gray",
                      fontSize: "small",
                      display: "flex",
                      right: "0px",
                    }}
                  >
                    <span style={{ color: "#b92b27" }}>
                      {answers.user.displayName
                        ? answers.user.displayName
                        : answers.user.email}{" "}
                      on{" "}
                      {new Date(answers.timestamp?.toDate()).toLocaleString()}
                    </span>
                  </span>
                </span>
              ) : (
                ""
              )}
            </p>
          ))}
        </div>
        
      </div>
      <div className="post__footer">
        <div className="post__footerAction">
        <h4>{likes}</h4><br></br>
                <button className="likes_button" onDoubleClick={handleLikes1}>  <ArrowUpwardOutlinedIcon /> </button>
                <h4>{dislikes}</h4><br></br>
          <button className="likes_button"><ArrowDownwardOutlinedIcon /></button>
        </div>
        
       
        <div className="post__footerLeft">
        
          
          
        </div>
      </div>
    </div>
  );
}

export default MyQuestionPost;