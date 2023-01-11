import React from "react";
import Feed from "./Feed.js";
import QHeader from "./QHeader";
import "./Curiosity.css";
import Sidebar from "./Sidebar";
import Category from "./Category";
import ContactUs from "./ContactUs";
import MyQuestions from "./MyQuestions";
import AboutUs from "./AboutUs";
import {BrowserRouter as Router,Switch,Route} from "react-router-dom";
function Curiosity() {
  return (
 
    <div className="Curiosity">
     <QHeader />
 
     <Router>
      <Sidebar />
      <Switch>
       

        <Route path='/Feed' exact component={Feed} />
        <Route path='/ContactUs' component={ContactUs} />
        <Route path='/MyQuestions' component={MyQuestions} />
        <Route path='/Category' component={Category} />
        <Route path ='/AboutUs' component={AboutUs}/>
        
       
       </Switch>
      </Router>
      </div>
  
  );
}

export default Curiosity;