"""empty message

Revision ID: e2fa62121621
Revises: dca1043624b8
Create Date: 2023-12-06 13:50:40.826019

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'e2fa62121621'
down_revision: Union[str, None] = 'dca1043624b8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # Add the lock_data column to the newpairs table
    op.add_column('newpairs', sa.Column(
        'is_honeypot', sa.Boolean, nullable=True))


def downgrade():
    # Remove the lock_data column from the newpairs table
    op.drop_column('newpairs', 'is_honeypot')
