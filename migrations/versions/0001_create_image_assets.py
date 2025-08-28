from alembic import op
import sqlalchemy as sa

revision = "0001_create_image_assets"
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    op.create_table(
        "image_assets",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("url", sa.String(length=512), nullable=False),
        sa.Column("label", sa.String(length=128), nullable=True),
        sa.Column("embedding_path", sa.String(length=512), nullable=True),
    )
    op.create_index("ix_image_assets_url", "image_assets", ["url"])

def downgrade():
    op.drop_index("ix_image_assets_url", table_name="image_assets")
    op.drop_table("image_assets")