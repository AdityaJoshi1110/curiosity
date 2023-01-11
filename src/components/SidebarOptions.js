import React, { useState } from 'react';
import * as AiIcons from 'react-icons/ai';

import { Link } from 'react-router-dom';
import { IconContext } from 'react-icons';

import "./SidebarOption.css";
import {SideBarData} from "./SideBarData.js"
function SidebarOptions() {

 
  const [sidebar] = useState(true);
 
  return (
  <>
     <IconContext.Provider value={{ color: '#fff', size: '25px' }}>
 
    <nav className={sidebar ? 'nav-menu active' : 'nav-menu'}>
      <ul className='nav-menu-items' >
     
    <div className="sidebarOptions">
      <div className="sidebarOption">
      
       {SideBarData.map((item, index) => {
          
           return (
                <li key={index} className={item.cName}>
                  <Link to={item.path}>
                    {item.icon}
                    <span>{item.title}</span>
                  </Link>
                </li>
              );
           
            })}
        
    </div>
    </div>
    </ul> 
     </nav>
      </IconContext.Provider>
    </>
  );
}

export default SidebarOptions;