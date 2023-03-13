from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# DATABASE_URL = "sqlite:///./mydatabase.db"\
DATABASE_URL =  "mysql+mysqlconnector://root@localhost:3306/mydatabase"

engine = create_engine(DATABASE_URL)

Session = sessionmaker(autocommit = False, autoflush= False, bind=engine)

base= declarative_base()

#creates database
def create_database():
    return base.metadata.create_all(bind = engine)