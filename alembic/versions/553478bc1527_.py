"""empty message

Revision ID: 553478bc1527
Revises: 675a477c20c4
Create Date: 2023-12-03 22:10:31.187717

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '553478bc1527'
down_revision: Union[str, None] = '675a477c20c4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('pairtrades',
                    sa.Column('id', sa.Integer, primary_key=True),
                    sa.Column('pair_address', sa.String(
                        length=50), nullable=False),
                    sa.Column('created_at', sa.Integer, nullable=False),
                    sa.Column('trade_details', sa.JSON, nullable=True),
                    sa.ForeignKeyConstraint(
                        ['pair_address'], ['newpairs.pair_address']),
                    sa.PrimaryKeyConstraint('id')
                    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('pairtrades')
    # ### end Alembic commands ###