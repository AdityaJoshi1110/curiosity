import firebase from "firebase";
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyCMO7-xjKUQAJ-dYIdfgUJq1c2ZGzjphp0",
  authDomain: "major-c9aa8.firebaseapp.com",
  projectId: "major-c9aa8",
  storageBucket: "major-c9aa8.appspot.com",
  messagingSenderId: "618587411629",
  appId: "1:618587411629:web:2326de2109e0227d7ca224",
  measurementId: "G-0V9FEGLXX4"
};
const firebaseApp = firebase.initializeApp(firebaseConfig);
const auth = firebase.auth();
const provider = new firebase.auth.GoogleAuthProvider();
const db = firebaseApp.firestore();

export { auth, provider };
export default db;
