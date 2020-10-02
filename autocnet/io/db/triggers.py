from sqlalchemy.schema import DDL

valid_geom_function = DDL("""
CREATE OR REPLACE FUNCTION validate_geom()
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

valid_geom_trigger = DDL("""
CREATE TRIGGER image_inserted
  BEFORE INSERT OR UPDATE
  ON images
  FOR EACH ROW
EXECUTE PROCEDURE validate_geom();
""")

valid_point_function = DDL("""
CREATE OR REPLACE FUNCTION validate_points()
  RETURNS trigger AS
$BODY$
BEGIN
 IF (SELECT COUNT(*)
	 FROM MEASURES 
   JOIN points ON points.id = NEW.pointid
	 WHERE pointid = NEW.pointid AND "measureIgnore" = False AND points."pointType" != 3) < 2
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

valid_point_trigger = DDL("""
CREATE TRIGGER active_measure_changes
  AFTER UPDATE
  ON measures
  FOR EACH ROW
EXECUTE PROCEDURE validate_points();
""")

ignore_image_function = DDL("""
CREATE OR REPLACE FUNCTION ignore_image()
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

ignore_image_trigger = DDL("""
CREATE TRIGGER image_ignored
  AFTER UPDATE
  ON images
  FOR EACH ROW
EXECUTE PROCEDURE ignore_image();
""")
