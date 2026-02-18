from __future__ import annotations

import random
from datetime import datetime, timedelta

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Giveaway, GiveawayParticipant, GiveawayWinner


class GiveawayService:
    def __init__(self, db_factory, admin_chat_ids: list[int], min_participants: int = 2) -> None:
        self.db_factory = db_factory
        self.admin_chat_ids = {int(x) for x in admin_chat_ids}
        self.min_participants = max(1, int(min_participants))
        self.rng = random.SystemRandom()

    def is_admin(self, chat_id: int) -> bool:
        return int(chat_id) in self.admin_chat_ids

    async def _active_giveaway(self, session: AsyncSession, group_chat_id: int) -> Giveaway | None:
        q = await session.execute(
            select(Giveaway)
            .where(Giveaway.chat_id == group_chat_id, Giveaway.status == "active")
            .order_by(Giveaway.id.desc())
            .limit(1)
        )
        return q.scalar_one_or_none()

    async def _last_winner(self, session: AsyncSession, group_chat_id: int) -> int | None:
        q = await session.execute(
            select(GiveawayWinner.user_chat_id)
            .where(GiveawayWinner.chat_id == group_chat_id)
            .order_by(desc(GiveawayWinner.won_at))
            .limit(1)
        )
        return q.scalar_one_or_none()

    async def _participant_ids(self, session: AsyncSession, giveaway_id: int) -> list[int]:
        q = await session.execute(
            select(GiveawayParticipant.user_chat_id)
            .where(GiveawayParticipant.giveaway_id == giveaway_id)
            .order_by(GiveawayParticipant.joined_at.asc())
        )
        return [int(x) for x in q.scalars().all()]

    async def _choose_winner(self, session: AsyncSession, group_chat_id: int, giveaway_id: int) -> tuple[int | None, str]:
        participants = await self._participant_ids(session, giveaway_id)
        if len(participants) < self.min_participants:
            return None, f"Need at least {self.min_participants} participants."

        last_winner = await self._last_winner(session, group_chat_id)
        eligible = [uid for uid in participants if uid != last_winner]
        if not eligible:
            eligible = participants
        return int(self.rng.choice(eligible)), "ok"

    async def start_giveaway(self, group_chat_id: int, admin_chat_id: int, duration_seconds: int, prize: str) -> dict:
        if not self.is_admin(admin_chat_id):
            raise PermissionError("Only configured admins can start giveaways.")
        duration = max(30, int(duration_seconds))

        async with self.db_factory() as session:
            existing = await self._active_giveaway(session, group_chat_id)
            if existing:
                raise RuntimeError("An active giveaway already exists in this chat.")

            now = datetime.utcnow()
            row = Giveaway(
                chat_id=group_chat_id,
                prize=(prize or "Prize").strip(),
                status="active",
                start_time=now,
                end_time=now + timedelta(seconds=duration),
                created_by_chat_id=admin_chat_id,
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return {
                "id": row.id,
                "chat_id": row.chat_id,
                "prize": row.prize,
                "status": row.status,
                "end_time": row.end_time.isoformat(),
            }

    async def join_active(self, group_chat_id: int, user_chat_id: int) -> dict:
        async with self.db_factory() as session:
            giveaway = await self._active_giveaway(session, group_chat_id)
            if not giveaway:
                raise RuntimeError("No active giveaway in this chat.")

            existing = await session.execute(
                select(GiveawayParticipant).where(
                    GiveawayParticipant.giveaway_id == giveaway.id,
                    GiveawayParticipant.user_chat_id == user_chat_id,
                )
            )
            row = existing.scalar_one_or_none()
            if not row:
                session.add(
                    GiveawayParticipant(
                        giveaway_id=giveaway.id,
                        user_chat_id=user_chat_id,
                    )
                )
                await session.commit()
            count_q = await session.execute(
                select(func.count(GiveawayParticipant.id)).where(GiveawayParticipant.giveaway_id == giveaway.id)
            )
            count = int(count_q.scalar_one() or 0)
            return {"giveaway_id": giveaway.id, "participants": count}

    async def _finalize(self, session: AsyncSession, giveaway: Giveaway) -> dict:
        winner_id, note = await self._choose_winner(session, giveaway.chat_id, giveaway.id)
        participants = await self._participant_ids(session, giveaway.id)

        if winner_id is None:
            giveaway.status = "ended_no_winner"
            giveaway.winner_user_id = None
            await session.commit()
            return {
                "giveaway_id": giveaway.id,
                "chat_id": giveaway.chat_id,
                "status": giveaway.status,
                "participants": len(participants),
                "winner_user_id": None,
                "prize": giveaway.prize,
                "note": note,
            }

        giveaway.status = "completed"
        giveaway.winner_user_id = winner_id
        session.add(
            GiveawayWinner(
                giveaway_id=giveaway.id,
                chat_id=giveaway.chat_id,
                user_chat_id=winner_id,
            )
        )
        await session.commit()
        return {
            "giveaway_id": giveaway.id,
            "chat_id": giveaway.chat_id,
            "status": giveaway.status,
            "participants": len(participants),
            "winner_user_id": winner_id,
            "prize": giveaway.prize,
            "note": "ok",
        }

    async def end_giveaway(self, group_chat_id: int, admin_chat_id: int) -> dict:
        if not self.is_admin(admin_chat_id):
            raise PermissionError("Only configured admins can end giveaways.")

        async with self.db_factory() as session:
            giveaway = await self._active_giveaway(session, group_chat_id)
            if not giveaway:
                raise RuntimeError("No active giveaway to end.")
            return await self._finalize(session, giveaway)

    async def reroll(self, group_chat_id: int, admin_chat_id: int) -> dict:
        if not self.is_admin(admin_chat_id):
            raise PermissionError("Only configured admins can reroll giveaways.")

        async with self.db_factory() as session:
            q = await session.execute(
                select(Giveaway)
                .where(Giveaway.chat_id == group_chat_id, Giveaway.status == "completed")
                .order_by(Giveaway.id.desc())
                .limit(1)
            )
            giveaway = q.scalar_one_or_none()
            if not giveaway:
                raise RuntimeError("No completed giveaway to reroll.")

            participants = await self._participant_ids(session, giveaway.id)
            if len(participants) < 2:
                raise RuntimeError("Need at least 2 participants to reroll.")

            old_winner = giveaway.winner_user_id
            eligible = [uid for uid in participants if uid != old_winner]
            if not eligible:
                raise RuntimeError("No alternate participant available for reroll.")
            new_winner = int(self.rng.choice(eligible))
            giveaway.winner_user_id = new_winner
            session.add(
                GiveawayWinner(
                    giveaway_id=giveaway.id,
                    chat_id=giveaway.chat_id,
                    user_chat_id=new_winner,
                )
            )
            await session.commit()
            return {
                "giveaway_id": giveaway.id,
                "chat_id": giveaway.chat_id,
                "status": giveaway.status,
                "participants": len(participants),
                "winner_user_id": new_winner,
                "previous_winner_user_id": old_winner,
                "prize": giveaway.prize,
                "note": "rerolled",
            }

    async def status(self, group_chat_id: int) -> dict:
        async with self.db_factory() as session:
            active = await self._active_giveaway(session, group_chat_id)
            if active:
                count_q = await session.execute(
                    select(func.count(GiveawayParticipant.id)).where(GiveawayParticipant.giveaway_id == active.id)
                )
                count = int(count_q.scalar_one() or 0)
                seconds_left = int(max((active.end_time - datetime.utcnow()).total_seconds(), 0))
                return {
                    "active": True,
                    "id": active.id,
                    "prize": active.prize,
                    "participants": count,
                    "seconds_left": seconds_left,
                    "end_time": active.end_time.isoformat(),
                }

            q = await session.execute(
                select(Giveaway)
                .where(Giveaway.chat_id == group_chat_id)
                .order_by(Giveaway.id.desc())
                .limit(1)
            )
            last = q.scalar_one_or_none()
            if not last:
                return {"active": False, "message": "No giveaway history in this chat."}
            return {
                "active": False,
                "id": last.id,
                "status": last.status,
                "prize": last.prize,
                "winner_user_id": last.winner_user_id,
                "end_time": last.end_time.isoformat(),
            }

    async def process_due_giveaways(self, notifier) -> int:
        now = datetime.utcnow()
        processed = 0
        async with self.db_factory() as session:
            q = await session.execute(
                select(Giveaway)
                .where(Giveaway.status == "active", Giveaway.end_time <= now)
                .order_by(Giveaway.end_time.asc())
            )
            due = list(q.scalars().all())
            for giveaway in due:
                payload = await self._finalize(session, giveaway)
                processed += 1
                if payload["winner_user_id"] is not None:
                    await notifier(
                        giveaway.chat_id,
                        (
                            f"Giveaway #{payload['giveaway_id']} ended.\n"
                            f"Winner: {payload['winner_user_id']}\n"
                            f"Prize: {payload['prize']}"
                        ),
                    )
                else:
                    await notifier(
                        giveaway.chat_id,
                        (
                            f"Giveaway #{payload['giveaway_id']} ended with no winner.\n"
                            f"{payload['note']}"
                        ),
                    )
        return processed
