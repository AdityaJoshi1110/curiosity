import React, { useEffect } from "react";
import { useDispatch, useSelector } from "react-redux";
import "./App.css";
import Login from "./components/auth/Login";
import Curiosity from "./components/Curiosity";
import { login, logout, selectUser } from "./features/userSlice";
import { auth } from "./firebase";
import '../node_modules/bootstrap/dist/css/bootstrap.min.css';
function App() {
  const user = useSelector(selectUser);
  const dispatch = useDispatch();
  useEffect(() => {
    auth.onAuthStateChanged((authUser) => {
      if (authUser) {
        dispatch(
          login({
            uid: authUser.uid,
            email: authUser.email,
            displayName: authUser.displayName,
            photo: authUser.photoURL,
          })
        );
      } else {
        dispatch(logout());
      }
      console.log(authUser);
    });
  }, [dispatch]);
  return <div className="App">{user ? <Curiosity /> : <Login />}</div>;
}

export default App;
