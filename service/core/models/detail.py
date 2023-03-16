import sqlalchemy as _sql
import sqlalchemy.orm as _orm

import service.core.config.db as _db

class Detail(_db.base):
    __tablename__ = "details"

    id = _sql.Column(_sql.Integer, primary_key=True)
    image_url = _sql.Column(_sql.String(255))
    # expectedReading = _sql.Column(_sql.String)
    expectedReading = _sql.Column(_sql.String(20))
    actualReading = _sql.Column(_sql.String(20))
    is_match = _sql.Column(_sql.Boolean, default=False)



