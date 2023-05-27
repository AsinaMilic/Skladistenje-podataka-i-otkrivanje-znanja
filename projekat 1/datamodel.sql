CREATE TABLE Hunts (
    hunt_id NUMBER(10) PRIMARY KEY,
    hunt_name VARCHAR2(255),
    hunt_description VARCHAR2(255),
    hunt_date DATE,
    location VARCHAR2(255),
    animals_hunted NUMBER(10),
    number_of_people NUMBER(10),
    duration NUMBER(10)   
);
CREATE TABLE Animals (
    animal_id NUMBER(10) PRIMARY KEY,
    hunt_id NUMBER(10),
    animal_name VARCHAR2(255),
    animal_description VARCHAR2(255),
    weight NUMBER(10),
    height NUMBER(10),
    price NUMBER(10),
    FOREIGN KEY (hunt_id) REFERENCES Hunts(hunt_id)
);

CREATE TABLE Hunt_Meetings (
    hunt_meeting_id NUMBER(10) PRIMARY KEY,
    hunt_id NUMBER(10),
    hunt_meeting_name VARCHAR2(255),
    hunt_meeting_description VARCHAR2(255),
    FOREIGN KEY (hunt_id) REFERENCES Hunts(hunt_id)
);

CREATE TABLE Members (
    member_id NUMBER(10) PRIMARY KEY,
    hunt_id NUMBER(10),
    name VARCHAR2(255),
    surname VARCHAR2(255),
    weight NUMBER(10),
    height NUMBER(10),
    age NUMBER(10),
    gender VARCHAR2(255),
    address VARCHAR2(255),
    email_address VARCHAR2(255),
    home_phone_number VARCHAR2(255),
    cell_mobile_phone_number VARCHAR2(255),
    FOREIGN KEY (hunt_id) REFERENCES Hunts(hunt_id)
);
CREATE TABLE Role (
    role_id NUMBER(10) PRIMARY KEY,
    role_name VARCHAR2(255),
    role_description VARCHAR2(255)
);
CREATE TABLE Hunt_Meeting_Participants (
    hunt_meeting_id NUMBER(10),
    member_id NUMBER(10),
    role_id NUMBER(10),
    PRIMARY KEY (hunt_meeting_id, member_id),
    FOREIGN KEY (hunt_meeting_id) REFERENCES Hunt_Meetings(hunt_meeting_id),
    FOREIGN KEY (member_id) REFERENCES Members(member_id)
);
CREATE TABLE Outcome (
    outcome_id NUMBER(10) PRIMARY KEY,
    hunt_meeting_id NUMBER(10),
    outcome_ending VARCHAR2(255),
    grade NUMBER(10),
    FOREIGN KEY (hunt_meeting_id) REFERENCES Hunt_Meetings(hunt_meeting_id)
);