CREATE TABLE SPECIE_DIM (
    SPECIE_DIM_ID NUMBER(*, 0) PRIMARY KEY,
    DATE_FROM DATE,
    DATE_TO DATE,
    SPECIE_ID NUMBER(*, 0),
    SPECIE_NAME VARCHAR2(255),
    SPECIE_DESCRIPTION VARCHAR2(255),
    PRICE_PER_KG NUMBER(*, 2),
    AVERAGE_LIFESPAN NUMBER(*, 0),
    DIET VARCHAR2(255),
    POPULATION_STATUS VARCHAR2(255),
    MAIN_PREDATOR VARCHAR2(255)
);

CREATE SEQUENCE SPECIE_DIM_S;

CREATE
OR REPLACE TRIGGER SPECIE_DIM_ON_INSERT BEFORE
INSERT
    ON SPECIE_DIM FOR EACH ROW BEGIN
SELECT
    SPECIE_DIM_S.NEXTVAL INTO :NEW.SPECIE_DIM_ID
FROM
    DUAL;
END;

CREATE TABLE DATE_DIM (
    DATE_DIM_ID NUMBER(*, 0) PRIMARY KEY,
    CATCH_DATE DATE,
    CATCH_DAY NUMBER(*, 0),
    CATCH_MONTH NUMBER(*, 0),
    CATCH_YEAR NUMBER(*, 0),
    CATCH_QUARTER NUMBER(*, 0)
);

CREATE SEQUENCE DATE_DIM_S;

CREATE
OR REPLACE TRIGGER DATE_DIM_ON_INSERT BEFORE
INSERT
    ON DATE_DIM FOR EACH ROW BEGIN
SELECT
    DATE_DIM_S.NEXTVAL INTO :NEW.DATE_DIM_ID
FROM
    DUAL;
END;

CREATE TABLE HUNTER_DIM (
    HUNTER_DIM_ID NUMBER(*, 0) PRIMARY KEY,
    DATE_FROM DATE,
    DATE_TO DATE,
    HUNTER_ID NUMBER(*, 0),
    NAME VARCHAR2(255),
    SURNAME VARCHAR2(255),
    WEIGHT NUMBER(*, 0),
    HEIGHT NUMBER(*, 0),
    AGE NUMBER(*, 0),
    GENDER VARCHAR2(255),
    ADDRESS VARCHAR2(255),
    EMAIL_ADRESS VARCHAR2(255),
    HOME_PHONE_NUMBER VARCHAR2(255),
    CELL_MOBILE_PHONE_NUMBER VARCHAR2(255),
    EXPERIENCE_YEARS NUMBER(*, 0)
);

CREATE SEQUENCE HUNTER_DIM_S;

CREATE
OR REPLACE TRIGGER HUNTER_DIM_ON_INSERT BEFORE
INSERT
    ON HUNTER_DIM FOR EACH ROW BEGIN
SELECT
    HUNTER_DIM_S.NEXTVAL INTO :NEW.HUNTER_DIM_ID
FROM
    DUAL;
END;

CREATE TABLE LOCATION_DIM (
    LOCATION_DIM_ID NUMBER(*, 0) PRIMARY KEY,
    DATE_FROM DATE,
    DATE_TO DATE,
    LOCATION_ID NUMBER(*, 0),
    LOCATION_NAME VARCHAR2(255),
    LOCATION_DESCRIPTION VARCHAR2(255),
    CLIMATE VARCHAR2(255),
    MAX_TEMPERATURE NUMBER(*, 2),
    MIN_TEMPERATURE NUMBER(*, 2),
    RAINY_DAYS NUMBER(*, 0),
    ALTITUDE NUMBER(*, 0),
    TERRAIN_TYPE VARCHAR2(255),
    VEGETATION_TYPE VARCHAR2(255),
    NEAREST_TOWN VARCHAR2(255)
);

CREATE SEQUENCE LOCATION_DIM_S;

CREATE
OR REPLACE TRIGGER LOCATION_DIM_ON_INSERT BEFORE
INSERT
    ON LOCATION_DIM FOR EACH ROW BEGIN
SELECT
    LOCATION_DIM_S.NEXTVAL INTO :NEW.LOCATION_DIM_ID
FROM
    DUAL;
END;

CREATE TABLE HUNT_DIM (
    HUNT_DIM_ID NUMBER(*, 0) PRIMARY KEY,
    DATE_FROM DATE,
    DATE_TO DATE,
    HUNT_ID NUMBER(*, 0),
    HUNT_NAME VARCHAR2(255),
    HUNTING_SEASON VARCHAR2(255),
    DURATION_HOURS NUMBER(*, 0)
);

CREATE SEQUENCE HUNT_DIM_S;

CREATE
OR REPLACE TRIGGER HUNT_DIM_ON_INSERT BEFORE
INSERT
    ON HUNT_DIM FOR EACH ROW BEGIN
SELECT
    HUNT_DIM_S.NEXTVAL INTO :NEW.HUNT_DIM_ID
FROM
    DUAL;
END;

CREATE TABLE PREYS_FACT (
    ANIMAL_ID NUMBER(*, 0),
    DATE_DIM_ID NUMBER(*, 0),
    SPECIE_DIM_ID NUMBER(*, 0),
    HUNTER_DIM_ID NUMBER(*, 0),
    LOCATION_DIM_ID NUMBER(*, 0),
    HUNT_DIM_ID NUMBER(*, 0),
    AGE NUMBER(*, 0),
    WEIGHT_KG NUMBER(*, 2),
    MEAT_QUALITY NUMBER(*, 2),
    TROPHY_POINTS NUMBER(*, 2),
    TOTAL_PRICE NUMBER(*, 2),
    PRIMARY KEY(
        ANIMAL_ID,
        DATE_DIM_ID,
        SPECIE_DIM_ID,
        HUNTER_DIM_ID,
        LOCATION_DIM_ID,
        HUNT_DIM_ID
    )
);