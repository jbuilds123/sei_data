"""empty message

Revision ID: 082c4e02a76c
Revises: 990e46c0e1ae
Create Date: 2023-11-16 11:27:47.981288

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '082c4e02a76c'
down_revision: Union[str, None] = '990e46c0e1ae'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # Add the lock_data column to the newpairs table
    op.add_column('newpairs', sa.Column('ohlcv_data', sa.JSON, nullable=True))


def downgrade():
    # Remove the lock_data column from the newpairs table
    op.drop_column('newpairs', 'ohlcv_data')
