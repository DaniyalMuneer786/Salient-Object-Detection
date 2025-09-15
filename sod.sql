-- phpMyAdmin SQL Dump
-- version 5.2.0
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Jun 17, 2025 at 02:00 PM
-- Server version: 10.4.27-MariaDB
-- PHP Version: 8.1.12

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `sod`
--

CREATE DATABASE IF NOT EXISTS `sod`;
USE `sod`;

-- --------------------------------------------------------

--
-- Table structure for table `admin`
--

CREATE TABLE `admin` (
  `Admin_ID` int(11) NOT NULL,
  `Name` varchar(100) DEFAULT NULL,
  `Email` varchar(100) DEFAULT NULL,
  `Password` varchar(100) DEFAULT NULL,
  `Salary` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;

--
-- Dumping data for table `admin`
--

INSERT INTO `admin` (`Admin_ID`, `Name`, `Email`, `Password`, `Salary`) VALUES
(50, 'Ahmer', 'a@gmail.com', '111', 100000);

-- --------------------------------------------------------

--
-- Table structure for table `feedback`
--

CREATE TABLE `feedback` (
  `Feedback_ID` int(11) NOT NULL,
  `Rating` int(11) DEFAULT NULL,
  `Feedback_Type` varchar(50) DEFAULT NULL,
  `Feedback_Text` text DEFAULT NULL,
  `User_ID` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;

--
-- Dumping data for table `feedback`
--

INSERT INTO `feedback` (`Feedback_ID`, `Rating`, `Feedback_Type`, `Feedback_Text`, `User_ID`) VALUES
(147, 3, 'general', 'No Big Issue!!!', 1);

-- --------------------------------------------------------

--
-- Table structure for table `image`
--

CREATE TABLE `image` (
  `image_id` int(11) NOT NULL,
  `uploaded_image` varchar(255) DEFAULT NULL,
  `file_type` varchar(10) DEFAULT NULL,
  `file_size` int(11) DEFAULT NULL,
  `user_id` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;

--
-- Dumping data for table `image`
--

INSERT INTO `image` (`image_id`, `uploaded_image`, `file_type`, `file_size`, `user_id`) VALUES
(1448, '1740990075_2.jpeg', 'jpeg', 5, 1),
(1680, '1740855465_sun_alfntqzssslakmss.jpg', 'jpg', 20, 1),
(1681, '1740854292_sun_ainjbonxmervsvpv.jpg', 'jpg', 52, 2),
(1749, '1740854292_sun_ainjbonxmervsvpv.jpg', 'jpg', 52, 2),
(1916, '1740989854_1.jpeg', 'jpeg', 8, 1),
(1996, '1740854292_sun_ainjbonxmervsvpv.jpg', 'jpg', 52, 2);

-- --------------------------------------------------------

--
-- Table structure for table `result`
--

CREATE TABLE `result` (
  `Result_ID` int(11) NOT NULL,
  `Result_data` text DEFAULT NULL,
  `Image_ID` int(11) DEFAULT NULL,
  `User_ID` int(11) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;

--
-- Dumping data for table `result`
--

INSERT INTO `result` (`Result_ID`, `Result_data`, `Image_ID`, `User_ID`) VALUES
(2097, 'Dominant Color: Blue, Type: jpg, Size: 19.87 KB', 1680, 1),
(2121, 'Dominant Color: Mixed, Type: jpeg, Size: 8.05 KB', 1916, 1),
(2321, 'Dominant Color: Mixed, Type: jpeg, Size: 5.34 KB', 1448, 1),
(2404, 'Dominant Color: Mixed, Type: jpg, Size: 51.96 KB', 1996, 2),
(2560, 'Dominant Color: Mixed, Type: jpg, Size: 51.96 KB', 1681, 2),
(2582, 'Dominant Color: Mixed, Type: jpg, Size: 51.96 KB', 1749, 2);

-- --------------------------------------------------------

--
-- Table structure for table `subscription`
--

CREATE TABLE `subscription` (
  `Subscription_ID` int(11) NOT NULL,
  `Plan_Type` enum('Basic','Pro','Enterprise') NOT NULL,
  `Start_Date` date,
  `End_Date` date DEFAULT NULL,
  `Status` enum('Active','Expired','Canceled') DEFAULT 'Active',
  `Upload_Limit` int(11) DEFAULT NULL,
  `Uploads_Used` int(11) DEFAULT 0,
  `Amount_Paid` decimal(10,2) DEFAULT NULL,
  `Payment_Method` enum('Credit Card','Debit Card','Stripe') NOT NULL,
  `User_ID` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;

--
-- Dumping data for table `subscription`
--

INSERT INTO `subscription` (`Subscription_ID`, `Plan_Type`, `Start_Date`, `End_Date`, `Status`, `Upload_Limit`, `Uploads_Used`, `Amount_Paid`, `Payment_Method`, `User_ID`) VALUES
(7, 'Enterprise', '2025-06-17', '2025-07-17', 'Active', NULL, 0, 99.99, 'Stripe', 1),
(19, 'Basic', '2025-06-17', '2025-07-17', 'Active', 100, 0, 9.99, 'Stripe', 1),
(74, 'Pro', '2025-06-17', '2025-07-17', 'Active', 500, 0, 29.99, 'Stripe', 1);

-- --------------------------------------------------------

--
-- Table structure for table `user_management`
--

CREATE TABLE `user_management` (
  `User_ID` int(11) NOT NULL,
  `Name` varchar(100) DEFAULT NULL,
  `Email` varchar(100) DEFAULT NULL,
  `Password` varchar(255) DEFAULT NULL,
  `Phone` varchar(20) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;

--
-- Dumping data for table `user_management`
--

INSERT INTO `user_management` (`User_ID`, `Name`, `Email`, `Password`, `Phone`) VALUES
(1, 'Daniyal Muneer', '211980031@gift.edu.pk', 'scrypt:32768:8:1$8IUfyp9Z7mJvhl76$25f3bfee71a89bbccfa6199b8597f23c6f634cacddcd6582cb40e2e7abc69fb679523352b0ff0452ed5a0bb7355ca9a037dd0d3b49ddd79fe8621d5dfccecab9', '03014858181'),
(2, 'Alyan', '211980033@gift.edu.pk', 'scrypt:32768:8:1$Jnz606eWqY8enBM8$f07196de02a1aa0d0468a6ea7f5206a48bfd00c1c7685eb8514a227a35883703d65dfedd3d6a4177176430f47b1faf4c222b3ae39ccdce4c7be97a69c55b96d6', '03014858181'),
(3, 'Hashim', '211980061@gift.edu.pk', 'scrypt:32768:8:1$SGNgp8uBaLYycWCT$52112a1eb3c514b15114e7415bd754148212e71795637db53bbb0ce5e9a9c749b01ecf66b658217a35e746323947529f0021498d66514f3318de880018245d12', '03014858181');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `admin`
--
ALTER TABLE `admin`
  ADD PRIMARY KEY (`Admin_ID`),
  ADD UNIQUE KEY `Email` (`Email`);

--
-- Indexes for table `feedback`
--
ALTER TABLE `feedback`
  ADD PRIMARY KEY (`Feedback_ID`),
  ADD KEY `User_ID` (`User_ID`);

--
-- Indexes for table `image`
--
ALTER TABLE `image`
  ADD PRIMARY KEY (`image_id`),
  ADD KEY `user_id` (`user_id`);

--
-- Indexes for table `result`
--
ALTER TABLE `result`
  ADD PRIMARY KEY (`Result_ID`),
  ADD KEY `Image_ID` (`Image_ID`),
  ADD KEY `User_ID` (`User_ID`);

--
-- Indexes for table `subscription`
--
ALTER TABLE `subscription`
  ADD PRIMARY KEY (`Subscription_ID`),
  ADD KEY `User_ID` (`User_ID`);

--
-- Indexes for table `user_management`
--
ALTER TABLE `user_management`
  ADD PRIMARY KEY (`User_ID`),
  ADD UNIQUE KEY `Email` (`Email`);

--
-- Constraints for dumped tables
--

--
-- Constraints for table `feedback`
--
ALTER TABLE `feedback`
  ADD CONSTRAINT `feedback_ibfk_1` FOREIGN KEY (`User_ID`) REFERENCES `user_management` (`User_ID`);

--
-- Constraints for table `image`
--
ALTER TABLE `image`
  ADD CONSTRAINT `image_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `user_management` (`User_ID`);

--
-- Constraints for table `result`
--
ALTER TABLE `result`
  ADD CONSTRAINT `result_ibfk_1` FOREIGN KEY (`Image_ID`) REFERENCES `image` (`image_id`),
  ADD CONSTRAINT `result_ibfk_2` FOREIGN KEY (`User_ID`) REFERENCES `user_management` (`User_ID`);

--
-- Constraints for table `subscription`
--
ALTER TABLE `subscription`
  ADD CONSTRAINT `subscription_ibfk_1` FOREIGN KEY (`User_ID`) REFERENCES `user_management` (`User_ID`);
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
