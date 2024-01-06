"""empty message

Revision ID: 675a477c20c4
Revises: 082c4e02a76c
Create Date: 2023-11-29 13:08:57.560771

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '675a477c20c4'
down_revision: Union[str, None] = '082c4e02a76c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # Add the lock_data column to the newpairs table
    op.add_column('newpairs', sa.Column('in_dex_time', sa.JSON, nullable=True))


def downgrade():
    # Remove the lock_data column from the newpairs table
    op.drop_column('newpairs', 'in_dex_time')
