import Select from "react-select";
import {Container,Col,Row} from 'react-bootstrap';
import React, { useEffect, useState } from "react";
import CuriosityBox from "./CuriosityBox";
import Post from "./Post";
import db from "../firebase";
import { useSelector } from "react-redux";
import { selectUser } from "../features/userSlice";
import "./Category.css"
function Category() {

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
  const[category,catValue]=useState("General");
  const [posts, setPosts] = useState([]);
  const user = useSelector(selectUser);
  useEffect(() => {
    if(user.uid){
    db.collection("questions").where("category","==",category)
      .orderBy("timestamp", "desc")
      .onSnapshot((snapshot) =>
        setPosts(
          snapshot.docs.map((doc) => ({
            id: doc.id,
            questions: doc.data(),
          }))
        )
      );}
  }, [user.uid,category]);
 
  const catHandler=e=>{
     catValue(e.label);
  }


  return (
    <>
  
   <div className="modal_category">
   <Select options={categoryList}  onChange={catHandler}>
            </Select>
    
    </div>
    <div className="category">
      <CuriosityBox />
      {
          
          posts.map(({ id, questions }) => (
        <Post
          key={id}
          Id={id}
          question={questions.question}
          imageUrl={questions.imageUrl}
          timestamp={questions.timestamp}
          users={questions.user}
          
        />
      ))}    
      </div> 
    </>
  );
}

export default Category;