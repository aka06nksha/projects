package com.SnakeGame;


import javax.swing.JFrame;

@SuppressWarnings("serial")
public class GameFrame extends JFrame {

   GameFrame(){
	
	   this.add(new GamePanel());
	   this.setTitle("snake");
	   this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	   this.setResizable(false);
	   this.pack();
	   this.setVisible(true);
	   this.setLocationRelativeTo(null);
	   
	
   }
     
}
