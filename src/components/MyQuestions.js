import React, { useEffect, useState } from "react";
import CuriosityBox from "./CuriosityBox";
import "./MyQuestions.css";
import MyQuestionPost from "./MyQuestionPost";
import db from "../firebase";
import { useSelector } from "react-redux";
import { selectUser } from "../features/userSlice";

function MyQuestions() {
  const [posts, setPosts] = useState([]);
  const user = useSelector(selectUser);

  useEffect(() => {
    if(user.uid){
    db.collection("questions").where("user.uid","==",user.uid)
      .orderBy("timestamp", "desc")
      .onSnapshot((snapshot) =>
        setPosts(
          snapshot.docs.map((doc) => ({
            id: doc.id,
            questions: doc.data(),
          }))
        )
      );}
  }, [user.uid]);

  return (
    <div className="MyQuestions">
      <CuriosityBox />
      {
          
          posts.map(({ id, questions }) => (
        <MyQuestionPost
          key={id}
          Id={id}
          question={questions.question}
          imageUrl={questions.imageUrl}
          timestamp={questions.timestamp}
          users={questions.user}
          
        />
      ))}
     
    </div>
  );
}

export default MyQuestions;