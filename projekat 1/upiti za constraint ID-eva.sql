DELETE FROM ANIMAL_DIM
WHERE animal_dim_id NOT IN (
  SELECT MIN(animal_dim_id)
  FROM ANIMAL_DIM
  GROUP BY animal_id
);

DELETE FROM OUTCOME_DIM
WHERE outcome_dim_id NOT IN (
  SELECT MIN(outcome_dim_id)
  FROM OUTCOME_DIM
  GROUP BY outcome_id
);


DELETE FROM ROLE_DIM
WHERE role_dim_id NOT IN (
  SELECT MIN(role_dim_id)
  FROM ROLE_DIM
  GROUP BY role_id
);

DELETE FROM HUNT_DIM
WHERE hunt_dim_id NOT IN (
  SELECT MIN(hunt_dim_id)
  FROM HUNT_DIM
  GROUP BY hunt_id
);

DELETE FROM MEMBER_DIM
WHERE member_dim_id NOT IN (
  SELECT MIN(member_dim_id)
  FROM MEMBER_DIM
  GROUP BY member_id
);