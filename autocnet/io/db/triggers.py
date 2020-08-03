from sqlalchemy.schema import DDL

def valid_geom_function(schema):
  return DDL(f"""
CREATE OR REPLACE FUNCTION {schema}.validate_geom()
  RETURNS trigger AS
$BODY$
  BEGIN
      NEW.geom = ST_MAKEVALID(NEW.geom);
      RETURN NEW;
    EXCEPTION WHEN OTHERS THEN
      NEW.ignore = true;
      RETURN NEW;
END;
$BODY$

LANGUAGE plpgsql VOLATILE -- Says the function is implemented in the plpgsql language; VOLATILE says the function has side effects.
COST 100; -- Estimated execution cost of the function.
""")

def valid_geom_trigger(schema):
  return DDL(f"""
CREATE TRIGGER image_inserted
  BEFORE INSERT OR UPDATE
  ON {schema}.images
  FOR EACH ROW
EXECUTE PROCEDURE validate_geom();
""")

def valid_point_function(schema):
  return DDL(f"""
CREATE OR REPLACE FUNCTION {schema}.validate_points()
  RETURNS trigger AS
$BODY$
BEGIN
 IF (SELECT COUNT(*)
	 FROM MEASURES
	 WHERE pointid = NEW.pointid AND "measureIgnore" = False) < 2
 THEN
   UPDATE points
     SET "pointIgnore" = True
	 WHERE points.id = NEW.pointid;
 ELSE
   UPDATE points
   SET "pointIgnore" = False
   WHERE points.id = NEW.pointid;
 END IF;

 RETURN NEW;
END;
$BODY$

LANGUAGE plpgsql VOLATILE -- Says the function is implemented in the plpgsql language; VOLATILE says the function has side effects.
COST 100; -- Estimated execution cost of the function.
""")

def valid_point_trigger(schema):
  return DDL(f"""
CREATE TRIGGER active_measure_changes
  AFTER UPDATE
  ON {schema}.measures
  FOR EACH ROW
EXECUTE PROCEDURE validate_points();
""")

def ignore_image_function(schema):
  return DDL(f"""
CREATE OR REPLACE FUNCTION {schema}.ignore_image()
  RETURNS trigger AS
$BODY$
BEGIN
 IF NEW.ignore
 THEN
   UPDATE measures
     SET "measureIgnore" = True
     WHERE measures.serialnumber = NEW.serial;
 END IF;

 RETURN NEW;
END;
$BODY$

LANGUAGE plpgsql VOLATILE -- Says the function is implemented in the plpgsql language; VOLATILE says the function has side effects.
COST 100; -- Estimated execution cost of the function.
""")

def ignore_image_trigger(schema):
  return DDL(f"""
CREATE TRIGGER image_ignored
  AFTER UPDATE
  ON {schema}.images
  FOR EACH ROW
EXECUTE PROCEDURE ignore_image();
""")
