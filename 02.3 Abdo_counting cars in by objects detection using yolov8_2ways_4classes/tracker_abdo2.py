import math

class Tracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        self.id_count = 0
        # Store how long an object hasn't been detected
        self.missing_tolerance = 10
        self.missing_frames = {}

    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x1, y1, x2, y2,label = rect
            cx = (x1 +x2 ) // 2
            cy = (y1 +y2 ) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                # Adjust this distance based on your needs (e.g. object speed)
                if dist < 40:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x1, y1, x2, y2, label, id])
                    same_object_detected = True
                    self.missing_frames[id] = 0  # Reset missing frame count
                    break

            # New object is detected, we assign a new ID to that object
            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x1, y1, x2, y2, label, self.id_count])
                self.missing_frames[self.id_count] = 0  # Initialize missing frame count
                self.id_count += 1

        # Clean the dictionary by center points to remove IDs not used anymore
        new_center_points = {}
        new_missing_frames = {}

        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center
            new_missing_frames[object_id] = self.missing_frames[object_id]

        # Remove objects that have been missing for too many frames
        for object_id in list(self.center_points):
            if object_id not in new_center_points:
                self.missing_frames[object_id] += 1
                if self.missing_frames[object_id] < self.missing_tolerance:
                    new_center_points[object_id] = self.center_points[object_id]
                    new_missing_frames[object_id] = self.missing_frames[object_id]

        self.center_points = new_center_points.copy()
        self.missing_frames = new_missing_frames.copy()

        return objects_bbs_ids
